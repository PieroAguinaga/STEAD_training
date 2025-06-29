import torch
import torch.nn.functional as F
import option
args = option.parse_args()
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import time

torch.autograd.set_detect_anomaly(True)

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        return torch.cdist(x, y, p=2)

    def forward(self, feats, margin=100.0):
        bs = feats.size(0)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]

        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)

        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)

        zero_tensor = torch.zeros_like(a_d_min, device=feats.device)
        a_d_min = margin - a_d_min
        a_d_min = torch.max(zero_tensor, a_d_min)

        return torch.mean(n_d_max) + torch.mean(a_d_min)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha=0.01):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, alpha * loss_triplet

def train(loader, model, optimizer, scheduler, device, epoch):
    model.train()
    loss_fn = Loss().to(device)

    preds, labels = [], []

    for step, (ninput, nlabel, ainput, alabel) in enumerate(loader):
        input = torch.cat((ninput, ainput), dim=0).to(device)
        label_batch = torch.cat((nlabel, alabel), dim=0).to(device)

        scores, feats = model(input)
        loss_ce, loss_triplet = loss_fn(scores.squeeze(), feats, label_batch)
        loss = loss_ce + loss_triplet

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(epoch * len(loader) + step)

        preds += scores.detach().cpu().sigmoid().squeeze().tolist()
        labels += label_batch.detach().cpu().tolist()

    auc_score = auc(*roc_curve(labels, preds)[:2])
    pr_score = auc(*precision_recall_curve(labels, preds)[1::-1])

    return loss.item()
