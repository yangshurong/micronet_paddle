from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print("pred_shape",pred.shape)
        # print('target_shape',target.shape)
        # print('target_shape1',target.view(1, -1).shape)
        # print('target_shape2',target.view(1, -1).expand_as(pred).shape)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            # print('k=',k,'correct_k',correct[:k].shape)

            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
