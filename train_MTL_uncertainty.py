import argparse
import os
import copy

import torch
from torch import nn
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models_ResNet_MLT import ResNetSR
from datasets_MLT import TrainDataset
from utils import AverageMeter, calc_psnr
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, device, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.device = device
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num),requires_grad=True, device=device))

    def forward(self, inputs, hr_loss, hq_loss):
     
        precision1 = torch.exp(-self.log_vars[0])
        loss_hr = torch.sum(precision1 * hr_loss + self.log_vars[0],-1)

        precision2 = torch.exp(-self.log_vars[1])
        loss_hq = torch.sum(precision2 * hq_loss + self.log_vars[1], -1)

        loss = loss_hr + loss_hq

        return loss, self.log_vars.data.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-train-file', type=str, required=True)
    parser.add_argument('--hq-train-file')
    parser.add_argument('--test-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    writer = SummaryWriter("runs/ResNetSR")
    
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(args.seed)

    print(device)
    print(torch.backends.cudnn.deterministic)
    print(torch.backends.cudnn.benchmark)

    model = ResNetSR().to(device=device,dtype=torch.float32)
    summary(model,(1,256,256))

    criterion = nn.MSELoss()
    mtl = MultiTaskLossWrapper(task_num=2,device=device,model = model)
    optimizer = optim.Adam([
        {'params': mtl.parameters() , 'lr': args.lr * 0.1}
    ], lr=args.lr)

    print('number of trainable parameters = : ' + str(sum(p.numel() for p in mtl.parameters() if p.requires_grad)))

    train_val_set = TrainDataset(args.hr_train_file,args.test_file,args.hq_train_file,'transpose')
    datasets = train_val_dataset(train_val_set)
    train_dataloader = DataLoader(dataset=datasets['train'],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataloader= DataLoader(dataset=datasets['val'],batch_size=1,shuffle=False)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

   
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataloader) - len(train_dataloader) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for batch_idx, data in enumerate(train_dataloader):
                inputs, hr_labels, hq_labels = data
        
               #load the data into the cuda:0 device 
                inputs = inputs.to(device=device,dtype=torch.float32)
                hr_labels = hr_labels.to(device=device,dtype=torch.float32)
                hq_labels = hq_labels.to(device=device,dtype=torch.float32)
                

                hr_preds, hq_preds = model(inputs)

                hr_loss = criterion(hr_preds, hr_labels)
                hq_loss = criterion(hq_preds, hq_labels)

                loss , log_vars = mtl(inputs,hr_loss,hq_loss)
                
                loss = loss.to(device)
                epoch_losses.update(loss.item(), len(inputs))
               
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                
        writer.add_scalar('training_loss',epoch_losses.avg,epoch)
        

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()

        hr_epoch_psnr = AverageMeter()
        hq_epoch_psnr = AverageMeter()

        hq_eval_losses  = AverageMeter()
        hr_eval_losses  = AverageMeter()
        eval_losses     = AverageMeter()

        print('[hr_weight, hq_weight] = ' + str(log_vars))

        for data in eval_dataloader:
            inputs, hr_labels, hq_labels = data

            inputs = inputs.to(device=device,dtype=torch.float32)
            hr_labels = hr_labels.to(device=device,dtype=torch.float32)
            hq_labels = hq_labels.to(device=device,dtype=torch.float32)

            with torch.no_grad():
                hr_preds,hq_preds = model(inputs)

            hr_eval_loss = criterion(hr_preds,hr_labels)
            hq_eval_loss = criterion(hq_preds,hq_labels)

            eval_losses.update(hr_eval_loss.item() + hq_eval_loss.item(), len(inputs))

            hr_epoch_psnr.update(calc_psnr(hr_preds,hr_labels), len(inputs))
            hq_epoch_psnr.update(calc_psnr(hq_preds,hq_labels), len(inputs))


        writer.add_scalar('eval_loss',eval_losses.avg,epoch)
        print('HR eval psnr: {:.2f}'.format(hr_epoch_psnr.avg))
        print('HQ eval psnr :{:.2f}'.format(hq_epoch_psnr.avg))
        writer.add_scalar('hr_psnr_eval',hr_epoch_psnr.avg,epoch)
        writer.add_scalar('hq_psnr_eval',hq_epoch_psnr.avg,epoch)

        hr_pred_grid=torchvision.utils.make_grid(hr_preds)
        hq_pred_grid=torchvision.utils.make_grid(hq_preds)
        writer.add_image('HR prediction epoch : ' + str(epoch),hr_pred_grid)
        writer.add_image('HQ prediction epoch : ' + str(epoch),hq_pred_grid)
        writer.close()

       # best epoch choice is dependant on what output is to be optimized  
        if hr_epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = hr_epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, hr_psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
