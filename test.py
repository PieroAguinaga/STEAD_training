from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args = option.parse_args()
from model import Model
from dataset import Dataset
from torchinfo import summary
import umap
import numpy as np
import sys
import time

MODEL_LOCATION = 'saved_models/'
MODEL_NAME = 'stead_model_base_ND2final'
MODEL_EXTENSION = '.pkl'

def test(dataloader, model, args, device='cuda', name="training", main=False):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_feats = []

    with torch.no_grad():
        start_time = time.time()
        for inputs in tqdm(dataloader, desc="Evaluando"):
            x, y = inputs
            x = x.to(device)
            y = y.to(device)

            scores, feats = model(x)
            scores = torch.sigmoid(scores).squeeze()

            all_preds.append(scores.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_feats.append(feats.detach().cpu())

        # Concatenar todos los batches
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_feats = torch.cat(all_feats).numpy()

        # Calcular métricas
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

        print(f"[⏱️ Tiempo] Evaluación completa: {time.time() - start_time:.2f}s")

        if main:
            reducer = umap.UMAP()
            reduced = reducer.fit_transform(all_feats)
            plt.figure()
            plt.scatter(reduced[all_labels == 0, 0], reduced[all_labels == 0, 1], c='tab:blue', label='Normal', marker='o')
            plt.scatter(reduced[all_labels == 1, 0], reduced[all_labels == 1, 1], c='tab:red', label='Anomaly', marker='*')
            plt.title('UMAP Embedding of Video Features')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.legend()
            plt.savefig(f"{name}_embed.png", bbox_inches='tight')
            plt.close()

        return roc_auc, pr_auc

if __name__ == '__main__':
    args = option.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_arch == 'base':
        model = Model()
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        print('Model architecture not recognized')
        sys.exit()

    test_loader = DataLoader(
        Dataset(args.dataset_path, RGB_list="./list/test.txt", test_mode=True),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = model.to(device)
    summary(model, (1, 192, 16, 10, 10))

    model.load_state_dict(torch.load(MODEL_LOCATION + MODEL_NAME + MODEL_EXTENSION))
    auc_score, pr_score = test(test_loader, model, args, device, name=MODEL_NAME, main=True)
    print(f"[📈 Resultados] AUC: {auc_score:.4f} | PR-AUC: {pr_score:.4f}")
