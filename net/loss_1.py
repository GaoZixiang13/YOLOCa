import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class yolox_loss(nn.Module):
    def __init__(self, input_shape, num_classes, device, label_smoothing=0, times=(8, 16, 32), cuda=True):
        super(yolox_loss, self).__init__()
        self.input_shape = input_shape
        # self.image_shape = image_shape
        self.num_classes = num_classes
        self.strides = [8, 16, 32]
        self.device = device
        self.times = times
        self.cuda = cuda
        self.dis = 2.5 * self.strides[0]
        self.cal0 = torch.tensor([0]).to(self.device)
        self.cal1 = torch.tensor([self.input_shape]).to(self.device)
        # self.ignore = noobj_ignore
        self.label_smoothing = label_smoothing
        # self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.style = style
        # self.sampling_size = sampling_size/self.image_shape

    #  将t中的所有值变动到[t_min, t_max]
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
        # w = w.squeeze(0)
        h = torch.exp(h) * stride

        prediction[..., 0], prediction[..., 1], prediction[..., 2], prediction[..., 3] = x, y, w, h
        # h = h.squeeze(0)
        # print(x.size())
        # print(y.size())
        # print(w.size())
        # print(h.size())
        # box = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)
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
        loss_reg, loss_conf, loss_cls, num_gts = 0, 0, 0, 0
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
            num_gt = gt_tures.size(0)
            # print(prediction_flat)
            # print(gt_tures)

            prediction_flat_x, prediction_flat_y = prediction_flat[..., 0], prediction_flat[..., 1] # [num_pred]
            # 中心点坐标在真实框中心点5*5大小框内的预测框
            gt_true_left = torch.maximum(gt_tures[..., 0] - self.dis, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0)) # [l_size, l_size]
            gt_true_right = torch.minimum(gt_tures[..., 0] + self.dis, self.cal1).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))
            gt_true_top = torch.maximum(gt_tures[..., 1] - self.dis, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))  # [l_size, l_size]
            gt_true_down = torch.minimum(gt_tures[..., 1] + self.dis, self.cal1).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))

            b_l = prediction_flat_x - gt_true_left
            b_r = gt_true_right - prediction_flat_x
            b_t = prediction_flat_y - gt_true_top
            b_b = gt_true_down - prediction_flat_y
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
            gt_tures_preds_mask = bbox_deltas.min(dim=-1).values >= 0.0

            # 中心点坐标在真实框内部的那些预测框
            gt_true_left_inbox = torch.maximum(gt_tures[..., 0] - gt_tures[..., 2]*0.5, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0)) # [l_size, l_size]
            gt_true_right_inbox = torch.minimum(gt_tures[..., 0] + gt_tures[..., 2]*0.5, self.cal1).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))
            gt_true_top_inbox = torch.maximum(gt_tures[..., 1] - gt_tures[..., 3]*0.5, self.cal0).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))  # [l_size, l_size]
            gt_true_down_inbox = torch.minimum(gt_tures[..., 1] + gt_tures[..., 3]*0.5, self.cal1).unsqueeze(-1).expand(gt_tures.size(0), prediction_flat.size(0))

            b_l = prediction_flat_x - gt_true_left_inbox
            b_r = gt_true_right_inbox - prediction_flat_x
            b_t = prediction_flat_y - gt_true_top_inbox
            b_b = gt_true_down_inbox - prediction_flat_y
            bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
            gt_tures_preds_mask_inbox = bbox_deltas.min(dim=-1).values >= 0.0

            # 这里取并集
            gt_tures_preds_mask = gt_tures_preds_mask | gt_tures_preds_mask_inbox
            gt_tures_preds_mask_and = gt_tures_preds_mask & gt_tures_preds_mask_inbox

            # print(torch.sum(gt_tures_preds_mask==True)/num_gt)
            # gt_tures_preds_mask_numpred = torch.sum(gt_tures_preds_mask, dim=0).type(torch.BoolTensor)
            # print(torch.sum(gt_tures_preds_mask_numpred==True).item())
            # print(torch.sum(gt_tures_preds_mask_numpred==False).item())
            # num_in_gt_tures = torch.sum()
            # print(torch.sum(gt_tures_preds_mask_numpred).item())
            prediction_decode = prediction_flat
            # 精筛
            for num, s_gt_trues in enumerate(gt_tures):
                prediction_or_filtered = prediction_flat[gt_tures_preds_mask]
                prediction_and_filtered = prediction_flat[gt_tures_preds_mask_and]
                # print(prediction_decode)
                s_gt_tures_ex = s_gt_trues.unsqueeze(-2).expand(s_gt_trues.size(0), prediction_or_filtered.size(0), 5 + self.num_classes) # [gt_tures_preds_mask] # [num_gt, t_num]
                iou = self.cal_iou(s_gt_tures_ex[..., :4], prediction_or_filtered[..., :4]) # [num_gt, num_pred]
                iou_cost = -torch.log(iou + 1e-8)
                cls_loss = self.BCEloss(s_gt_tures_ex[..., 5:], prediction_or_filtered[..., 5:]).squeeze(-1) # [num_gt, num_pred]
                '''
                3, 10为超参数，根据不同的数据集取值
                '''
                # print(iou.size())
                # print(cls_loss.size())
                cost_function = (cls_loss + 3 * iou_cost + 100000.0*~gt_tures_preds_mask_and)

                topk = torch.clamp(torch.sum(torch.topk(iou*gt_tures_preds_mask.int(), k=10, dim=-1).values, dim=-1).type(torch.IntTensor), min=1)
                # top_ious, top_ious_indices = torch.sort(iou, dim=-1, descending=True)
                # topk = torch.sum(top_ious[..., :min(10, top_ious_indices.size(0))], dim=-1).type(torch.int)

                gt_tures_preds_last_mask = torch.zeros_like(gt_tures_preds_mask).type(torch.BoolTensor).to(self.device)

                # idx0 = gt_tures_preds_mask[num].nonzero().squeeze(1)
                # cost_sorted = torch.argsort(cost_function[num, gt_tures_preds_mask[num]], dim=-1)
                # idx = cost_sorted[:topk[num]]
                _, idx = torch.topk(cost_function[num], k=min(topk[num].item(), max(1, torch.sum(gt_tures_preds_mask_and[num]==True).item())), largest=False)
                gt_tures_preds_last_mask[num, idx] = True

            # 去除重叠样本
            gt_tures_preds_last_mask_num = torch.sum(gt_tures_preds_last_mask.type(torch.IntTensor), dim=0) # [num_pred]
            # cost_f_sorted_values, cost_f_sorted_indices = torch.sort(cost_function, dim=-1)
            overlap_mask = gt_tures_preds_last_mask_num > 1 # [num_pred]
            cost_min_gt_indices = torch.argmin(cost_function[..., overlap_mask], dim=0) # [num_overlap]
            gt_tures_preds_last_mask[..., overlap_mask] = False
            gt_tures_preds_last_mask[cost_min_gt_indices, overlap_mask] = True

            gt_tures_preds_last_mask_calconf = torch.sum(gt_tures_preds_last_mask, dim=0).type(torch.BoolTensor).to(self.device)
            # for num, s_gt_trues in enumerate(gt_tures):
            # 计算正样本的损失
            # print(torch.sum(gt_tures_preds_last_mask_calconf).item())
            # num_gt = gt_tures.size(0)
            # print(torch.sum(gt_tures_preds_last_mask==True)/num_gt)
            # print(gt_tures_preds_last_mask)

            loss_conf += torch.sum(self.BCEloss(prediction_flat[..., 4], gt_tures_preds_last_mask_calconf.int()))
            loss_cls += torch.sum(cls_loss[gt_tures_preds_last_mask])
            loss_reg += torch.sum(1 - iou[gt_tures_preds_last_mask] ** 2)
            num_gts += num_gt

        reg_weight = 5.0
        num_gts = max(num_gts, 1)
        loss = reg_weight * loss_reg/num_gts + loss_cls/num_gts + loss_conf/num_gts

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
