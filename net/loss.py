import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class yolox_loss(nn.Module):
    def __init__(self, input_shape, num_classes, device, label_smoothing=0, times=(8, 16, 32), cuda=True):
        super(yolox_loss, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.strides = (8, 16, 32)
        self.device = device
        self.times = times
        self.cuda = cuda
        self.dis = 2.5 * self.strides[0]
        self.cal0 = torch.tensor([0]).to(self.device)
        # self.cal1 = torch.tensor([1]).to(self.device)
        self.cal_input = torch.tensor([self.input_shape]).to(self.device)
        self.label_smoothing = label_smoothing

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCEloss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def decode(self, prediction, l_size, stride):
        '''
        :param x: tensor: [l_size, l_size]
        :param y:
        :param w:
        :param h:
        :return: [..., 4]
        '''
        x, y, w, h = prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3]
        ix = torch.arange(l_size).repeat(l_size, 1).to(self.device)
        iy = torch.arange(l_size).repeat(l_size, 1).t().to(self.device)
        x = (ix+x) * stride
        y = (iy+y) * stride
        w = torch.exp(w) * stride
        h = torch.exp(h) * stride

        prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3] = x, y, w, h

        return prediction

    def forward(self, predictions, targets):
        '''
        :param preds: torch tensor --> [batch_size, 3*(5+num_classes), l_size, l_size]
        :param target: torch tensor -->
        :return:
        '''
        # print(torch.sum(target[..., 0] == 1))
        l_sizes = [self.input_shape//time for time in self.times]
        bs = predictions[0].size(0)
        loss_reg, loss_conf, loss_cls, num_fgs = 0, 0, 0, 0
        # return torch.sum(predictions[0][..., 4])
        preds = []
        for l, prediction_s in enumerate(predictions):
            l_size = l_sizes[l]
            prediction_s = prediction_s.view(bs, 5+self.num_classes, l_size, l_size).permute(0, 2, 3, 1).contiguous()
            prediction_s[..., :2] = torch.sigmoid(prediction_s[..., :2])
            prediction_s[..., 4:] = torch.sigmoid(prediction_s[..., 4:])
            prediction_s = self.decode(prediction_s, l_size, self.strides[l]).view(bs, l_size*l_size, 5+self.num_classes)
            preds.append(prediction_s)
        predictions_concat = torch.cat(preds, dim=1)

        for b in range(bs):
            prediction_flat = predictions_concat[b]
            target_mask = targets[b][..., 4] == 1
            gt_tures = targets[b][target_mask]  # .expand(target_mask.size(0), prediction)
            if gt_tures.size(0) == 0:
                loss_conf += torch.sum(self.BCEloss(prediction_flat[..., 4], self.cal0))
                continue
            # print(prediction_flat)
            # print(gt_tures)

            prediction_flat_x, prediction_flat_y = prediction_flat[..., 0], prediction_flat[..., 1] # [num_pred]
            # 中心点坐标在真实框中心点5*5大小框内的预测框
            gt_true_left = torch.maximum(gt_tures[..., 0] - self.dis, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0)) # [l_size, l_size]
            gt_true_right = torch.minimum(gt_tures[..., 0] + self.dis, self.cal_input).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))
            gt_true_top = torch.maximum(gt_tures[..., 1] - self.dis, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))  # [l_size, l_size]
            gt_true_down = torch.minimum(gt_tures[..., 1] + self.dis, self.cal_input).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))

            b_l = prediction_flat_x - gt_true_left
            b_r = gt_true_right - prediction_flat_x
            b_t = prediction_flat_y - gt_true_top
            b_b = gt_true_down - prediction_flat_y
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
            gt_tures_preds_mask_center = bbox_deltas.min(dim=-1).values >= 0.0
            del gt_true_left, gt_true_right, gt_true_top, gt_true_down

            # 中心点坐标在真实框内部的那些预测框
            gt_true_left_inbox = torch.maximum(gt_tures[..., 0] - gt_tures[..., 2]*0.5, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0)) # [l_size, l_size]
            gt_true_right_inbox = torch.minimum(gt_tures[..., 0] + gt_tures[..., 2]*0.5, self.cal_input).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))
            gt_true_top_inbox = torch.maximum(gt_tures[..., 1] - gt_tures[..., 3]*0.5, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))  # [l_size, l_size]
            gt_true_down_inbox = torch.minimum(gt_tures[..., 1] + gt_tures[..., 3]*0.5, self.cal_input).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))

            b_l = prediction_flat_x - gt_true_left_inbox
            b_r = gt_true_right_inbox - prediction_flat_x
            b_t = prediction_flat_y - gt_true_top_inbox
            b_b = gt_true_down_inbox - prediction_flat_y
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
            gt_tures_preds_mask_inbox = bbox_deltas.min(dim=-1).values >= 0.0
            del b_l, b_r, b_t, b_b, bbox_deltas, prediction_flat_x, prediction_flat_y
            del gt_true_left_inbox, gt_true_right_inbox, gt_true_top_inbox, gt_true_down_inbox

            # 这里取并集
            # tmp = torch.tensor(gt_tures_preds_mask)
            gt_tures_preds_mask_or = gt_tures_preds_mask_center | gt_tures_preds_mask_inbox
            gt_tures_preds_mask_and = gt_tures_preds_mask_center & gt_tures_preds_mask_inbox
            del gt_tures_preds_mask_center, gt_tures_preds_mask_inbox

            prediction_decode = prediction_flat
            # ------------------------------------------------------------------------------------------
            # 统一处理的代码
            gt_tures_ex = gt_tures.unsqueeze(-2).expand(gt_tures.size(0), prediction_decode.size(0), 5 + self.num_classes) # [gt_tures_preds_mask] # [num_gt, t_num]
            iou = self.cal_iou(gt_tures_ex[..., :4], prediction_decode[..., :4]) # [num_gt, num_pred]
            iou_cost = -torch.log(iou + 1e-8)
            cls_loss = self.BCEloss(gt_tures_ex[..., 5:], prediction_decode[..., 5:]).sum(-1) # [num_gt, num_pred]
            # gt_pred_zs = torch.minimum(gt_tures_ex[..., :2]-0.5*gt_tures_ex[..., 2:4], prediction_decode[..., :2]-0.5*prediction_decode[..., 2:4])
            # gt_pred_yx = torch.maximum(gt_tures_ex[..., :2]+0.5*gt_tures_ex[..., 2:4], prediction_decode[..., :2]+0.5*prediction_decode[..., 2:4])
            # gt_pred_c2 = (gt_pred_yx - gt_pred_zs).pow(2).sum(-1)
            dis2_gt_center = (gt_tures_ex[..., :2] - prediction_decode[..., :2]).pow(2).sum(-1).pow(0.5)
            cost_dis = dis2_gt_center / (self.dis * (2**0.5))
            # del gt_pred_zs, gt_pred_yx, gt_pred_c2, dis2_gt_center
            '''
            3, 10为超参数，根据不同的数据集取值
            '''
            # cost_function = (cls_loss + 3 * iou_cost + 100000.0*~gt_tures_preds_mask_and)
            cost_function = (cls_loss + 3*iou_cost + cost_dis + 100000.0*~gt_tures_preds_mask_and)

            topk = torch.clamp(torch.sum(torch.topk(iou*gt_tures_preds_mask_or.int(), k=10, dim=-1).values, dim=-1).type(torch.IntTensor), min=1)

            gt_tures_preds_last_mask = torch.zeros_like(gt_tures_preds_mask_or).type(torch.BoolTensor).to(self.device)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 精筛
            # cost_function, topk, gt_tures_preds_mask_and, gt_tures_preds_last_mask = cost_function.tolist(), topk.tolist(), gt_tures_preds_mask_and.tolsit(), gt_tures_preds_last_mask.tolist()
            for num in range(gt_tures.size(0)):
                _, idx = torch.topk(cost_function[num], k=min(topk[num].item(), max(1, torch.sum(gt_tures_preds_mask_and[num]==True).item())), largest=False)
                gt_tures_preds_last_mask[num, idx] = True
            # cost_function, topk, gt_tures_preds_mask_and, gt_tures_preds_last_mask = torch.tenosr(cost_function), torch.tenosr(topk), torch.tenosr(gt_tures_preds_mask_and), torch.tenosr(gt_tures_preds_last_mask)
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # 去除重叠样本
            gt_tures_preds_last_mask_num = torch.sum(gt_tures_preds_last_mask.type(torch.IntTensor), dim=0).to(self.device) # [num_pred]
            # cost_f_sorted_values, cost_f_sorted_indices = torch.sort(cost_function, dim=-1)
            overlap_mask = gt_tures_preds_last_mask_num > 1 # [num_pred]
            cost_min_gt_indices = torch.argmin(cost_function[..., overlap_mask], dim=0) # [num_overlap]
            gt_tures_preds_last_mask[..., overlap_mask] = False
            gt_tures_preds_last_mask[cost_min_gt_indices, overlap_mask] = True

            gt_tures_preds_last_mask_calconf = torch.sum(gt_tures_preds_last_mask, dim=0).type(torch.BoolTensor).to(self.device)

            # cls_mask = (~gt_tures_preds_last_mask & gt_tures_preds_mask_or).sum(0)

            loss_conf += torch.sum(self.BCEloss(prediction_decode[..., 4], gt_tures_preds_last_mask_calconf.int()))
            loss_cls += torch.sum(cls_loss[gt_tures_preds_last_mask])# + torch.sum(self.BCEloss(prediction_decode[cls_mask][..., 5:], self.cal0))
            loss_reg += torch.sum(1 - iou[gt_tures_preds_last_mask] ** 2)
            num_fg = torch.sum(gt_tures_preds_last_mask.sum(0) == True).item()
            num_fgs += num_fg
            # ------------------------------------------------------------------------------------------

        reg_weight = 5.0
        num_fgs = max(num_fgs, 1)
        loss = reg_weight * loss_reg/num_fgs + loss_cls/num_fgs + loss_conf/num_fgs

        return loss

    def cal_iou(self, boxes1, box2):
        '''
        :param boxes1:
        :param box2:
        :return:
        '''
        zsxy = torch.maximum(boxes1[..., 0:2] - boxes1[..., 2:4]/2, box2[..., 0:2] - box2[..., 2:4]/2)
        yxxy = torch.minimum(boxes1[..., 0:2] + boxes1[..., 2:4]/2, box2[..., 0:2] + box2[..., 2:4]/2)
        wh = torch.maximum(yxxy - zsxy, torch.zeros_like(yxxy))
        Intersection = wh[..., 0] * wh[..., 1]
        Union = boxes1[..., 2] * boxes1[..., 3] + box2[..., 2] * box2[..., 3] - Intersection

        return Intersection/torch.clamp(Union, 1e-6)

# ------------------------------------------------------------------------------------------
# 挨个处理试试会不会更快
# for num, s_gt_trues in enumerate(gt_tures):
#     prediction_this_gt_or = prediction_decode[gt_tures_preds_mask_or]
#     prediction_this_gt_and = prediction_decode[gt_tures_preds_mask_and]
#
#     s_gt_trues_ex_or = s_gt_trues.unsqueeze(-2).expand(1, prediction_this_gt_or.size(0), 5 + self.num_classes)
#     iou_t = self.cal_iou(s_gt_trues_ex_or[..., :4], prediction_this_gt_or[..., :4])  # [num_gt, num_pred]
#     del s_gt_trues_ex_or
#
#     s_gt_trues_ex_and = s_gt_trues.unsqueeze(-2).expand(1, prediction_this_gt_and.size(0), 5 + self.num_classes)
#     iou = self.cal_iou(s_gt_trues_ex_and[..., :4], prediction_this_gt_and[..., :4])
#     iou_cost = -torch.log(iou + 1e-8)
#
#     # cls_loss = self.BCEloss(s_gt_trues_ex[..., 5:], prediction_this_gt[..., 5:]).squeeze(
#     #     -1)  # [num_gt, num_pred]
#
#     gt_pred_zs = torch.minimum(s_gt_trues_ex_and[..., :2] - 0.5 * s_gt_trues_ex_and[..., 2:4],
#                                prediction_this_gt_and[..., :2] - 0.5 * prediction_this_gt_and[..., 2:4])
#     gt_pred_yx = torch.maximum(s_gt_trues_ex_and[..., :2] + 0.5 * s_gt_trues_ex_and[..., 2:4],
#                                prediction_this_gt_and[..., :2] + 0.5 * prediction_this_gt_and[..., 2:4])
#     gt_pred_c2 = (gt_pred_yx - gt_pred_zs).pow(2).sum(-1)
#     dis2_gt_center = (s_gt_trues_ex_and[..., :2] - prediction_this_gt_and[..., :2]).pow(2).sum(-1)
#     cost_dis = dis2_gt_center / gt_pred_c2
#     del gt_pred_zs, gt_pred_yx, gt_pred_c2, dis2_gt_center
#
#     cost_function = (iou_cost + cost_dis)
#
#     topk = torch.clamp(
#         torch.sum(torch.topk(iou_t, k=10, dim=-1).values, dim=-1).type(
#             torch.IntTensor), min=1)
#
#     gt_tures_preds_last_mask = torch.zeros_like(prediction_this_gt_and).type(torch.BoolTensor).to(
#         self.device)
#     _, idx = torch.topk(cost_function[num], k=min(topk[num].item(), max(1, torch.sum(
#         prediction_this_gt_and.size(0)).item())), largest=False)
#     gt_tures_preds_last_mask[num, idx] = True
#
#     # 去除重叠样本
#     gt_tures_preds_last_mask_num = torch.sum(gt_tures_preds_last_mask.type(torch.IntTensor),
#                                              dim=0)  # [num_pred]
#     # cost_f_sorted_values, cost_f_sorted_indices = torch.sort(cost_function, dim=-1)
#     overlap_mask = gt_tures_preds_last_mask_num > 1  # [num_pred]
#     cost_min_gt_indices = torch.argmin(cost_function[..., overlap_mask], dim=0)  # [num_overlap]
#     gt_tures_preds_last_mask[..., overlap_mask] = False
#     gt_tures_preds_last_mask[cost_min_gt_indices, overlap_mask] = True
#
#     gt_tures_preds_last_mask_calconf = torch.sum(gt_tures_preds_last_mask, dim=0).type(torch.BoolTensor)
#     # cls_mask = (~gt_tures_preds_last_mask & gt_tures_preds_mask_or).sum(0)
#
#     loss_conf += torch.sum(self.BCEloss(prediction_this_gt_and[gt_tures_preds_last_mask_calconf][..., 4], self.cal1))
#     loss_cls += torch.sum(self.BCEloss(prediction_this_gt_and[gt_tures_preds_last_mask_calconf][..., 5:], self.cal1)) # + torch.sum(self.BCEloss(prediction_decode[cls_mask][..., 5:], self.cal0))
#     loss_reg += torch.sum(1 - iou[gt_tures_preds_last_mask] ** 2)
#     num_fg = torch.sum(gt_tures_preds_last_mask.sum(0) == True).item()
#     num_fgs += num_fg
# ------------------------------------------------------------------------------------------


