import torch

def GT_aleatoric_loss(loss, pred_map, gt_map, logvar):
    term1 = 0.5 * torch.exp(-logvar) * loss(gt_map, pred_map)
    term2 = 0.5 * logvar
    return (term1 + term2).sum()


def bayes_aleatoric_loss(loss, preds, targets, logvar, prob_list):
    term1 = 0.5 * torch.exp(-logvar) * loss(prob_list, targets, preds)
    term2 = 0.5 * logvar
    return (term1 + term2).sum()
