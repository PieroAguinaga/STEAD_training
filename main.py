from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option
import time
import pandas as pd
args = option.parse_args()

from model import Model
from dataset import Dataset
from train import train
from test import test
import os
import random
import sys
from save_results import save_results
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


if __name__ == '__main__':
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")

    train_loader = DataLoader(
        Dataset(dataset_path=args.dataset_path, RGB_list="./list/train.txt", test_mode=False),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )

    test_loader = DataLoader(
        Dataset(dataset_path=args.dataset_path, RGB_list="./list/test.txt", test_mode=True),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )

    if args.model_arch == 'base':
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch in ['fast', 'tiny']:
        model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate, ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        print("Model architecture not recognized")
        sys.exit()

    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("✅ Checkpoint preentrenado cargado")

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.max_epoch * num_steps,
        cycle_mul=1.,
        lr_min=args.lr * 0.2,
        warmup_lr_init=args.lr * 0.01,
        warmup_t=args.warmup * num_steps,
        cycle_limit=20,
        t_in_epochs=False,
        warmup_prefix=True,
        cycle_decay=0.95,
    )

    from train import Loss
    loss_fn = Loss().to(device)

    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}

    itr = 1
    best_auc = 0

    for step in tqdm(range(0, args.max_epoch), total=args.max_epoch, dynamic_ncols=True):
        epoch_start = time.perf_counter()
        model.train()

        # Entrenamiento
        t0 = time.perf_counter()
        for _, (ninput, nlabel, ainput, alabel) in enumerate(train_loader):
            input = torch.cat((ninput, ainput), dim=0).to(device)
            label_batch = torch.cat((nlabel, alabel), dim=0).to(device)

            optimizer.zero_grad()
            with autocast():
                scores, feats = model(input)
                loss_ce, loss_triplet = loss_fn(scores.squeeze(), feats, label_batch)
                loss = loss_ce + loss_triplet
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        t1 = time.perf_counter()
        print(f"[⏱️ Tiempo] Entrenamiento epoca {step}: {t1 - t0:.2f} s")

        # Scheduler
        t0 = time.perf_counter()
        scheduler.step(step + 1)
        t1 = time.perf_counter()
        print(f"[⏱️ Tiempo] Scheduler epoca {step}: {t1 - t0:.2f} s")

        # Evaluación cada 5 épocas o la última
        if step % 5 == 0 or step == args.max_epoch - 1:
            t0 = time.perf_counter()
            auc, pr_auc = test(test_loader, model, args, device)
            t1 = time.perf_counter()
            print(f"[⏱️ Tiempo] Evaluación epoca {step}: {t1 - t0:.2f} s")

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR"].append(pr_auc)

            # Guardar mejor modelo
            t0 = time.perf_counter()
            if auc > best_auc:
                best_auc = auc
                best_model_path = os.path.join("./ckpt/", f"{args.model_name}final.pkl")
                torch.save(model.state_dict(), best_model_path)
                print(f"[💾 Guardado] Modelo mejorado guardado en epoca {step} con AUC={best_auc:.4f}")
            t1 = time.perf_counter()
            print(f"[⏱️ Tiempo] Guardado modelo epoca {step}: {t1 - t0:.2f} s")

        epoch_end = time.perf_counter()
        print(f"⏱️ Tiempo total epoca {step}: {epoch_end - epoch_start:.2f} s\n")

        itr += 1

    # Guardar CSV de resultados
    df = pd.DataFrame(test_info)
    df.to_csv(f"./ckpt/{args.model_name}_metrics.csv", index=False)
    print(f"📁 Resultados guardados en ./ckpt/{args.model_name}_metrics.csv")
