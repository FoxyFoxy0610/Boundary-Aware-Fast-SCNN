"""Custom losses."""
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss', 'BoundaryAwareLoss']

def boundary_map(mask, dilation_ratio=0.02):
    """
    mask: tensor (H, W), binary {0,1}
    dilation_ratio: relative boundary thickness (e.g., 0.02 of diagonal length)
    """
    mask = mask.cpu().numpy().astype(np.uint8)
    h, w = mask.shape
    diag_len = np.sqrt(h**2 + w**2)
    dilation = max(1, int(dilation_ratio * diag_len))

    dilated = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
    eroded = cv2.erode(mask, np.ones((dilation, dilation), np.uint8))
    boundary = dilated - eroded
    return torch.from_numpy(boundary).to(torch.float32)


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-Aware Loss (BIoU) integration.
    This loss combines standard segmentation loss with Boundary Intersection over Union (BIoU)
    to explicitly optimize for sharp and accurate boundaries, which is crucial for
    agricultural plots and lane detection.
    """
    def __init__(self, base_loss_fn, dilation_ratio=0.02, eps=1e-6):
        super(BoundaryAwareLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.dilation_ratio = dilation_ratio
        self.eps = eps

    def forward(self, predict, target, biou_ratio=0.0, weight=None):
        # 1. Compute Base Loss (e.g. CrossEntropy or OHEM)
        contour_loss = self.base_loss_fn(predict, target)
        
        if biou_ratio <= 0.0:
            return contour_loss, 0.0

        # 2. Compute BIoU Loss
        if isinstance(predict, tuple) or isinstance(predict, list):
            pred_logits = predict[0]
        else:
            pred_logits = predict

        num_classes = pred_logits.shape[1]
        device = pred_logits.device
        pred = torch.argmax(pred_logits, dim=1)  # [B, H, W]

        biou_total_loss = 0.0
        mbiou_batch_sum = 0.0
        batch_size = pred.shape[0]

        for b in range(batch_size):
            pred_mask = pred[b]
            gt_mask = target[b]

            biou_sum = 0.0
            count = 0
            for c in range(1, num_classes):
                pred_c = (pred_mask == c).to(torch.uint8)
                gt_c   = (gt_mask == c).to(torch.uint8)

                if gt_c.sum() == 0 and pred_c.sum() == 0:
                    continue

                boundary_pred = boundary_map(pred_c, self.dilation_ratio).to(device)
                boundary_gt   = boundary_map(gt_c, self.dilation_ratio).to(device)

                inter = (boundary_pred * boundary_gt).sum()
                union = boundary_pred.sum() + boundary_gt.sum() - inter
                biou = (inter + self.eps) / (union + self.eps)
                
                biou_sum += biou
                count += 1

            if count > 0:
                biou_total_loss += (1 - biou_sum / count)
                mbiou_batch_sum += (biou_sum / count)

        biou_total_loss /= batch_size
        mbiou = mbiou_batch_sum / batch_size if batch_size > 0 else 0.0

        loss = (1 - biou_ratio) * contour_loss + biou_ratio * biou_total_loss
        return loss, mbiou


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=True, class_weights=None, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight and class_weights is not None:
            weight = torch.FloatTensor(class_weights)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        elif use_weight:
            # Default fallback if class_weights not provided but use_weight is True
            weight = torch.FloatTensor([1, 1])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            pass # Keep all
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
