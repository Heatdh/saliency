import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import SaliconDataset
from loss import *
import cv2
from utils import blur, AverageMeter
import wandb
import json



class train_simpleNet():
    def __init__(self, dict_args, cfg): 
        self.args = dict_args
        self.cfg =  cfg.split(".")[0]
        print("Training with config: ", self.cfg)
        print("Training with args: ", self.args)
        

    def config_Wb(self):
        """ Configure wb
        """
        unique_id=wandb.util.generate_id()
        wandb.init(id=unique_id,name=self.cfg, project='Saliency', entity='heatdh', config=self.args)

        #wandb.init(id= args.enc_model, project="Saliency", entity="heatdh")#, config=args,reinit=True)

    def dump_args(args):
        with open(os.path.join('cfgs/default_cfg.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    def Sal_Dataloader(self):
        train_img_dir = self.args["dataset_dir"] + "images/train/"
        train_gt_dir = self.args["dataset_dir"] + "maps/train/"
        train_fix_dir = self.args["dataset_dir"] + "fixations/train/"

        val_img_dir = self.args["dataset_dir"] + "images/val/"
        val_gt_dir = self.args["dataset_dir"] + "maps/val/"
        val_fix_dir = self.args["dataset_dir"] + "fixations/val/"
        train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
        val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
        train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids,".jpg")
        val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids,".jpg")
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["no_workers"])
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=self.args["no_workers"])

    def loss_func(self,pred_map, gt, fixations):
        self.loss = torch.FloatTensor([0.0]).cuda()
        criterion = nn.L1Loss()
        if self.args["kldiv"]:
            self.loss += self.args["kldiv_coeff"] * kldiv(pred_map, gt)
        if self.args["cc"]:
            self.loss += self.args["cc_coeff"] * cc(pred_map, gt)
        if self.args["nss"]:
            self.loss += self.args["nss_coeff"] * nss(pred_map, fixations)
        if self.args["l1"]:
            self.loss += self.args["l1_coeff"] * criterion(pred_map, gt)
        if self.args["sim"]:
            self.loss += self.args["sim_coeff"] * similarity(pred_map, gt)
        return self.loss

    
    def load_encoder(self):
        
        if self.args["enc_model"] == "pnas":
            print("PNAS Model")
            from model import PNASModel
            self.model = PNASModel(train_enc=bool(self.args["train_enc"]), load_weight=self.args["load_weight"])

        elif self.args["enc_model"] == "densenet":
            print("DenseNet Model")
            from model import DenseModel
            self.model = DenseModel(train_enc=bool(self.args["train_enc"]), load_weight=self.args["load_weight"])

        elif self.args["enc_model"] == "resnet":
            print("ResNet Model")
            from model import ResNetModel
            self.model = ResNetModel(train_enc=bool(self.args["train_enc"]), load_weight=self.args["load_weight"])
            
        elif self.args["enc_model"] == "vgg":
            print("VGG Model")
            from model import VGGModel
            self.model = VGGModel(train_enc=bool(self.args["train_enc"]), load_weight=self.args["load_weight"])

        elif self.args["enc_model"] == "mobilenet":
            print("Mobile NetV2")
            from model import MobileNetV2
            self.model = MobileNetV2(train_enc=bool(self.args["train_enc"]), load_weight=self.args["load_weight"])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self,model, optimizer, loader, epoch, device): # not wanting to access them through self to leave the parameters visible
        model.train()
        tic = time.time()
        
        total_loss = 0.0
        cur_loss = 0.0
        wandb.watch(model)
        for idx, (img, gt, fixations) in enumerate(loader):
            img = img.to(device)
            gt = gt.to(device)
            fixations = fixations.to(device)
            
            optimizer.zero_grad()
            pred_map = model(img)
            assert pred_map.size() == gt.size()
            loss = self.loss_func(pred_map, gt, fixations)
            loss.backward()
            total_loss += loss.item()
            cur_loss += loss.item()
            
            optimizer.step()
            if idx%self.args["log_interval"]==(self.args["log_interval"]-1):
                print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/self.args["log_interval"], (time.time()-tic)/60))
                cur_loss = 0.0
                sys.stdout.flush()
            wandb.log({"loss":loss})
        wandb.log({"avg_loss":total_loss/len(loader)})
        
        print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
        sys.stdout.flush()
        #wandb.log({"avg_train_loss":total_loss/(idx+1)})
        return total_loss/len(loader)

    def set_optimizer(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters())) 

        if self.args["optim"]=="Adam":
            self.optimizer = torch.optim.Adam(params, lr=self.args["lr"])
        if self.args["optim"]=="Adagrad":
            self.optimizer = torch.optim.Adagrad(params, lr=self.args["lr"])
        if self.args["optim"]=="SGD":
            self.optimizer = torch.optim.SGD(params, lr=self.args["lr"], momentum=0.9)
        if self.args["lr_sched"]:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args["step_size"], gamma=0.1)

    def validate(self,model, loader, epoch, device):
        model.eval()
        tic = time.time()
        total_loss = 0.0
        cc_loss = AverageMeter()
        kldiv_loss = AverageMeter()
        nss_loss = AverageMeter()
        sim_loss = AverageMeter()
        
        for (img, gt, fixations) in loader:
            img = img.to(device)
            gt = gt.to(device)
            fixations = fixations.to(device)
            
            pred_map = model(img)

            # Blurring
            blur_map = pred_map.cpu().squeeze(0).clone().numpy()
            blur_map = blur(blur_map).unsqueeze(0).to(device)
            
            cc_loss.update(cc(blur_map, gt))    
            kldiv_loss.update(kldiv(blur_map, gt))    
            nss_loss.update(nss(fixations, gt)) # was previously blur map
            sim_loss.update(similarity(blur_map, gt))    
        wandb.log({"cc_loss":cc_loss.avg})
        wandb.log({"kldiv_loss":kldiv_loss.avg})
        wandb.log({"nss_loss":nss_loss.avg})
        wandb.log({"sim_loss":sim_loss.avg})
        print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
        sys.stdout.flush()
        
        return cc_loss.avg

    def train_model(self):
        for epoch in range(0, self.args["no_epochs"]):
            loss = self.train(self.model, self.optimizer, self.train_loader, epoch, self.device)
            
            with torch.no_grad():
                cc_loss = self.validate(self.model, self.val_loader, epoch, self.device)
                if epoch == 0 :
                    best_loss = cc_loss
                if best_loss <= cc_loss:
                    best_loss = cc_loss
                    print('[{:2d},  save, {}]'.format(epoch, self.args["model_val_path"]))
                    if torch.cuda.device_count() > 1:    
                        torch.save(self.model.module.state_dict(), self.args["model_val_path"])
                    else:
                        torch.save(self.model.state_dict(), self.args["model_val_path"])
                print()

            if self.args["lr_sched"]:
                self.scheduler.step()

