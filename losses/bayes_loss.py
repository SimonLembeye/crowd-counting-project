import torch
from torch.nn import Module


class BayesLoss(Module):
    def __init__(self, use_background, device):
        super(BayesLoss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0

        """
            - prob list semble être la listes des p(yn|xm) ie la contribution du pixel xm sur la n-ieme tête
            (les lignes de cette matrice sont de taille 4096 = 64*64)
            - pre density est la prédiction de la densité (sortie du réseau) - de taille 64x64 ici
            - target list a pour longueur le nombre de têtes - correspond aux E[cn] "réel" (le calcul reste un mystère)
            - On obtient les E[cn] estimées grâce à un produit terme à terme de prob_list et pre_density
        """


        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss
