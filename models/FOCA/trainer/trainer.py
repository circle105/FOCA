from models import ad_predict
from models.reasonable_metric import tsad_reasonable, reasonable_accumulator
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.tools import adjust_learning_rate, cal_accuracy, adjustment
from .early_stopping import EarlyStopping

from merlion.evaluate.anomaly import ScoreType

from torch import optim
from datetime import datetime

import os
import time
import math
import warnings
import numpy as np
import pandas as pd
import yaml
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, device, config, idx, seed, load_based_model=True):
        super(Trainer, self).__init__()
        self.device = device
        self.config = config

        self.model = model
        if load_based_model:
            self._set_network()
        
        folder = important_args(self.config)
        self.path = f"./log/foca/{self.config.dataset}/{folder}/seed{seed}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print(config, folder=self.path)
        self.save_ckpt = True
        self.save_ckpt = False

        self.early_stopping = EarlyStopping(self.path, idx, self.save_ckpt)
        self.idx = idx

        self.R = torch.tensor(0.0, device=self.device)  # hypersphere radius R
        self.c = torch.zeros(self.config.hidden_size, device=self.device)  # hypersphere center c
        self.nu = config.nu
        self.objective = config.objective
    
    def _save_model(self):
        ckptpath = os.path.join(self.path, "best_network", str(self.idx).zfill(3) + "_best_network.pth")
        os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
        state = {"model_state_dict": self.model.state_dict(), "R": self.R, "c": self.c}
        torch.save(state, ckptpath)
        return ckptpath

    def _load_pretrain_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.R = checkpoint["R"]
        self.c = checkpoint["c"]

    def _set_network(self):
        import importlib

        module = importlib.import_module("models.UniTS.network.model")
        based_model = module.base_Model(self.config).to(self.device)

        pretrain_weight_path = "./ckpt/based/units/units_x32_pretrain_checkpoint.pth"
        print("loading pretrained units model:", pretrain_weight_path)
        if "pretrain_checkpoint.pth" in pretrain_weight_path:
            state_dict = torch.load(pretrain_weight_path, map_location="cpu")["student"]
            ckpt = {}
            for k, v in state_dict.items():
                if not ("cls_prompts" in k):
                    ckpt[k] = v
        else:
            ckpt = torch.load(pretrain_weight_path, map_location="cpu")
        ckpt_copy = {}

        for k, v in ckpt.items():
            name = k[7:]  # remove `module.`
            ckpt_copy[name] = v
        msg = based_model.load_state_dict(ckpt_copy, strict=False)
        net_dict = self.model.state_dict()
        units_net_dict = based_model.state_dict()

        units_net_dict = {k: v for k, v in units_net_dict.items() if k in net_dict}
        net_dict.update(units_net_dict)
        self.model.load_state_dict(net_dict)

    def train(self, train_dl, val_dl, test_dl):
        print(self.path, folder=self.path)

        lora_lr = self.config.lora_lr
        final_layer_lr = self.config.final_layer_lr
        base_params = [param for name, param in self.model.named_parameters() if 'lora'  in name]
        optimizer = optim.Adam([
            {'params': base_params, 'lr': lora_lr},
            {'params': self.model.out_projection.parameters(), 'lr': final_layer_lr}
            ], lr=final_layer_lr, 
            betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True)
        scaler = NativeScaler()     
        self.model.train()
        
        for epoch in range(self.config.train_epochs):
            self.choose_training_parts(epoch=epoch)
            if epoch < self.config.change_center_epoch:
                self.c = self.center_c(train_dl, device=self.device, eps=self.config.center_eps)
            epoch_time = datetime.now()
            train_loss, self.R, norm_value = self.train_one_epoch(optimizer, train_dl, self.c, self.R, scaler, epoch)
            epoch_time = datetime.now() - epoch_time
            val_target, val_score_origin, val_loss = self.test(val_dl, epoch)
            test_target, test_score_origin, test_loss = self.test(test_dl, epoch)
            scheduler.step(train_loss)
            print("Epoch {}\t|train_loss {:.4f}\t|norm_value {:.4f}\t|val_loss {:.4f}\t|test_loss {:.4f}\t| cost {}".format(epoch, train_loss, norm_value, test_loss, val_loss, epoch_time), folder=self.path)

            state = {"model_state_dict": self.model.state_dict(), "R": self.R, "c": self.c}
            if self.config.early_stopping and self.config.dataset == "UCR":
                val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, self.config.threshold_determine, self.config.detect_nu)
                test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target, test_score_origin, self.config.threshold_determine, self.config.detect_nu)
                score_reasonable = tsad_reasonable(test_target, predict, time_step=1)
                self.early_stopping(score_reasonable, test_affiliation, test_rpa_score, test_rpa_score.f1(ScoreType.RevisedPointAdjusted), val_score_origin, test_score_origin, state)
                if self.early_stopping.early_stop:
                    print("Early stopping", folder=self.path)
                    break
            elif self.config.early_stopping and  ( "WADI" in self.config.dataset):
                self.early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, state)
                if self.early_stopping.early_stop:
                    print("Early stopping", folder=self.path)
                    break
        print("Training is Done!" + "#" * 15, folder=self.path)

        if self.config.early_stopping and self.config.dataset == "UCR":
            score_reasonable = self.early_stopping.best_score_reasonable
            test_score_origin = self.early_stopping.best_predict2
            test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target, test_score_origin, self.config.threshold_determine, self.config.detect_nu)
        elif self.config.early_stopping and  "WADI" in self.config.dataset:
            val_score_origin = self.early_stopping.best_predict1
            test_score_origin = self.early_stopping.best_predict2
            print("best loss: {:.4f}".format(self.early_stopping.best_indicator))
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, self.config.threshold_determine, self.config.detect_nu)
            test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target, test_score_origin, self.config.threshold_determine, self.config.detect_nu)
            score_reasonable = reasonable_accumulator(1, 0)
            val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
            val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
            val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
            print(f'Valid affiliation-metrics\t precision: {val_affiliation["precision"]:2.4f}  | \trecall: {val_affiliation["recall"]:2.4f}', folder=self.path)
            print(f"Valid RAP\tF1: {val_f1:2.4f}  | \t precision: {val_precision:2.4f}  | \t recall: {val_recall:2.4f}", folder=self.path)
        else:
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, self.config.threshold_determine, self.config.detect_nu)
            test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target, test_score_origin, self.config.threshold_determine, self.config.detect_nu)
            score_reasonable = reasonable_accumulator(1, 0)
            val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
            val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
            val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
            print(f'Valid affiliation-metrics\t precision: {val_affiliation["precision"]:2.4f}  | \trecall: {val_affiliation["recall"]:2.4f}', folder=self.path)
            print(f"Valid RAP\tF1: {val_f1:2.4f}  | \t precision: {val_precision:2.4f}  | \t recall: {val_recall:2.4f}", folder=self.path)
            if self.save_ckpt:
                print(self._save_model(), folder=self.path)
        
        # result
        print(f"{self.config.dataset} {self.idx:03d}", folder=self.path)
        print(f'Test affiliation-metrics\t F1: {2*test_affiliation["precision"]*test_affiliation["recall"]/(test_affiliation["precision"]+test_affiliation["recall"]):2.4f}\t precision: {test_affiliation["precision"]:2.4f}  | \trecall: {test_affiliation["recall"]:2.4f}', folder=self.path)
        print(f"*Test RAP\t F1: {test_rpa_score.f1(ScoreType.RevisedPointAdjusted):2.4f}  | \tprecision: {test_rpa_score.precision(ScoreType.RevisedPointAdjusted):2.4f}  | \trecall: {test_rpa_score.recall(ScoreType.RevisedPointAdjusted):2.4f}", folder=self.path)
        print(f"Test PA\t F1: {test_pa_score.f1(ScoreType.PointAdjusted):2.4f}  | \tprecision: {test_pa_score.precision(ScoreType.PointAdjusted):2.4f}  | \trecall: {test_pa_score.recall(ScoreType.PointAdjusted):2.4f}", folder=self.path)
        print(f"Test PW\t F1: {test_pw_score.f1(ScoreType.PointAdjusted):2.4f}  | \tprecision: {test_pw_score.precision(ScoreType.PointAdjusted):2.4f}  | \trecall: {test_pw_score.recall(ScoreType.PointAdjusted):2.4f}", folder=self.path)

        return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict

    def train_one_epoch(self, optimizer, train_loader, center, length, scaler, epoch):
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        max_norm = self.config.clip_grad
        norm_value = 0
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_x, target = data.float().to(self.device).permute(0, 2, 1), target.long()
            with torch.cuda.amp.autocast():
                features = self.model(batch_x, None, None, None, task_id=0, task_name="anomaly_detection")
                loss, score = self.calculate_loss_score(features, center, length, epoch)
                if (self.objective == "soft-boundary") and (epoch >= self.config.freeze_length_epoch):
                    length = torch.tensor(self.get_radius(score, self.nu), device=self.device)
                loss_scale = 1.0
                total_loss.append(loss.item())
            norm_value = scaler(loss * loss_scale, optimizer, clip_grad=max_norm, parameters=self.model.parameters(), create_graph=False, update_grad=True)
            optimizer.zero_grad()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss, length, norm_value

    def choose_training_parts(self, epoch):
        prompt_tune = False
        lora_tune = False
        if epoch >= self.config.lora_tune_epoch:
            lora_tune = True

        for name, param in self.model.named_parameters():
            if prompt_tune and "prompt_token" in name or "mask_prompt" in name or "cls_prompt" in name or "mask_token" in name or "cls_token" in name or "category_token" in name:
                param.requires_grad = True
            elif lora_tune and "lora" in name:
                param.requires_grad = True
            elif "out_projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def test(self, test_dl, epoch=-1):
        total_loss, total_f1, total_precision, total_recall = [], [], [], []
        all_target, all_predict = [], []
        self.model.eval()
        with torch.no_grad():
            for data, target in test_dl:
                inputs, target = data.float().to(self.device).permute(0, 2, 1), target.long()
                features = self.model(inputs, None, None, None, task_id=0, task_name="anomaly_detection")
                loss, score = self.calculate_loss_score(features, self.c, self.R, -1)
                total_loss.append(loss.item())
                predict = score.detach().cpu().numpy()
                
                all_target.extend(target)
                all_predict.extend(predict)

        total_loss = torch.tensor(total_loss).mean()  # average loss
        all_target = np.array(all_target)
        return all_target, all_predict, total_loss

    def center_c(self, train_loader, device, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = self.c
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                # get the inputs of the batch
                batch_x, target = data.float().to(self.device).permute(0, 2, 1), target.long()
                outputs = self.model(batch_x, None, None, None, task_id=0, task_name="anomaly_detection")
                c += torch.sum(outputs, dim=0)
                n_samples += outputs.shape[0]

        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def calculate_loss_score(self, feature1, center, length, epoch):
        # normalize feature vectors
        center = center.unsqueeze(0)
        center = F.normalize(center, dim=1)
        feature1 = F.normalize(feature1, dim=1)
        distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
        distance1 = 1 - distance1

        # Prevent model collapse
        sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
        sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1.0 - sigma_aug1))
        loss_sigam = torch.mean(sigma_loss1) # v(Q)

        # The Loss function that representations reconstruction
        score = distance1
        if self.objective == "soft-boundary":
            diff1 = score - length
            loss_oc = length + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
        else:
            loss_oc = torch.mean(score)
        loss = self.config.omega1 * loss_oc + self.config.omega2 * loss_sigam
        return loss, score

    def get_radius(self, dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        dist = dist.reshape(-1)
        return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)

def important_args(args):
    args_important_simply = "{}_{}_nu{}_o2{}_r{}_a{}_l1{}_l2{}_wd{}_loraEpo{}_te{}_ce{}_es{}_hz{}".format(
        args.window_size,args.time_step,
        args.nu, args.omega2,
        args.lora_rank, args.lora_alpha, args.lora_lr,args.final_layer_lr,
        args.weight_decay, args.lora_tune_epoch,
        args.train_epochs, args.change_center_epoch,args.early_stopping,args.hidden_size)
    return  args_important_simply

def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = " ".join(map(str, args))
        if "file" not in kwargs or kwargs["file"] is None:
            sys.stdout.write(text + "\n")
        else:
            kwargs["file"].write(text + "\n")

        if "folder" in kwargs and kwargs["folder"]:
            with open("{}/{}_debug.log".format(kwargs.get("folder", ""), cur_time), "a") as log_file:
                log_file.write(text + "\n")
        if "folder" in kwargs:
            del kwargs["folder"]
        if "file" in kwargs:
            del kwargs["file"]

    return wrapper


print = custom_print_decorator(print)
cur_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
