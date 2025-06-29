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
args=option.parse_args()
from model import Model
from dataset import Dataset

from train import train
from test import test
import datetime
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

    args=option.parse_args()
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    device = torch.device('cuda')


    # DO NOT SHUFFLE, shuffling is handled by the Dataset class and not the DataLoader

    train_loader = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/train.txt", test_mode=False), batch_size=args.batch_size)

    test_loader = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/test.txt", test_mode=True), batch_size=args.batch_size)

    test_loader_train = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/train.txt", test_mode=True), batch_size=args.batch_size)
    
    test_loader_L = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/test_L.txt", test_mode=True), batch_size=args.batch_size)
    
    test_loader_M = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/test_M.txt", test_mode=True), batch_size=args.batch_size)

    test_loader_S = DataLoader(Dataset(dataset_path=args.dataset_path, RGB_list= "./list/test_S.txt", test_mode=True), batch_size=args.batch_size)

    if args.model_arch == 'base':
        model = Model(dropout = args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = Model(dropout = args.dropout_rate, attn_dropout=args.attn_dropout_rate, ff_mult = 1, dims = (32,32), depths = (1,1))
    else:
        print("Model architecture not recognized")
        sys.exit()
    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial= args.max_epoch * num_steps,
            cycle_mul=1.,
            lr_min=args.lr * 0.2,
            warmup_lr_init=args.lr * 0.01,
            warmup_t=args.warmup * num_steps,
            cycle_limit=20,
            t_in_epochs=False,
            warmup_prefix=True,
            cycle_decay = 0.95,
        )

    test_info = {"epoch": [], "test_AUC": [], "test_PR":[]}

    itr = 1
    best_auc = 0
    for step in tqdm(
            range(0, args.max_epoch),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        cost = train(train_loader, model, optimizer, scheduler, device, step)

        scheduler.step(step + 1)

        auc_train, pr_auc = test(test_loader_train, model, args, device)

        auc_test_L, pr_auc = test(test_loader_L, model, args, device)

        auc_test_M, pr_auc = test(test_loader_M, model, args, device)

        auc_test_S, pr_auc = test(test_loader_S, model, args, device)


        auc, pr_auc = test(test_loader, model, args, device)

        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)

        save_results(args.model_name,itr,auc_train= auc_train, auc_test=auc,auc_test_l= auc_test_L,auc_test_m=auc_test_M, auc_test_s=auc_test_S)

        # Guardado general por época
        #torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-x3d.pkl'.format(step))
        #save_best_record(test_info, os.path.join(savepath + "/", '{}-step.txt'.format(step)))
    
        # Guardar mejor modelo
        if auc > best_auc:
            best_auc = auc
            best_model_path = os.path.join("./ckpt/", f"{args.model_name}final.pkl")
            torch.save(model.state_dict(), best_model_path)
            #with open(os.path.join(savepath, "best_result.txt"), 'w') as f:
            #    f.write(f"Epoch: {step}\n")
            #    f.write(f"Best AUC: {best_auc:.6f}\n")

        
        itr +=1



#python git_main.py  --comment vf --batch_size 16 --model_name STEAD_FAST_XS --model_arch fast --dataset_path features_x3d_xs 