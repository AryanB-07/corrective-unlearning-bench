import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning, distill_kl_loss, compute_accuracy
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
import json
import os

class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.pretrain_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.pretrain_iters*1.25, warmup_epochs=self.opt.pretrain_iters//100)
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        self.scaler = GradScaler()

    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        self.model.cuda()

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss

    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for (images, target, infgt) in tqdm.tqdm(loader):
            images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return

    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for (images, target) in tqdm.tqdm(loader):
                with autocast():
                    images, target = images.cuda(), target.cuda()
                    output = self.model(images) if self.prenet is None else self.model(self.prenet(images))
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')

        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return

    def store_final_acc(self):
        """ Helper to store the best val acc before & after unlearning """
        if hasattr(self, 'best_top1'):
            if not hasattr(self, 'best_pretrain_val_top1'):
                self.best_pretrain_val_top1 = self.best_top1
                print(f"[store_final_acc] Saved pretrain acc: {self.best_pretrain_val_top1*100:.2f}%")
            else:
                self.best_unlearn_val_top1 = self.best_top1
                print(f"[store_final_acc] Saved unlearn acc: {self.best_unlearn_val_top1*100:.2f}%")

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        # === Pretraining phase ===
        while self.curr_step < self.opt.pretrain_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start

        self.store_final_acc()  # Save pretrain acc

        # === Unlearning phase ===
        self.curr_step = 0
        self.best_top1 = 0  # Reset for unlearning
        while self.curr_step < self.opt.unlearn_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start

        self.store_final_acc()  # Save unlearn acc

        # === Final summary ===
        print("\n====== FINAL RESULTS SUMMARY ======")
        print(f"Before Unlearning (Test Acc): {self.best_pretrain_val_top1*100:.2f}%")
        print(f"After Unlearning  (Test Acc): {self.best_unlearn_val_top1*100:.2f}%")
        print("====================================")

        # Save summary to JSON
        summary = {
            "Pretrain_Val_Acc": round(self.best_pretrain_val_top1*100, 2),
            "Unlearn_Val_Acc": round(self.best_unlearn_val_top1*100, 2),
        }
        os.makedirs(self.opt.save_dir, exist_ok=True)
        with open(f"{self.opt.save_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return


# ✅ Keep ApplyK as is, with your fallback!
class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(model, k=self.opt.k, model_name=self.opt.model)
        model = unlearn_func(model=model, method=self.opt.unlearn_method, factor=self.opt.factor, device=self.opt.device)
        self.model = model
        self.prenet = prenet
        self.model.cuda()
        if self.prenet is not None:
            self.prenet.cuda().eval()

    def divide_model(self, model, k, model_name):
        if int(k) == -1:
            net = model
            prenet = None
            return prenet, net

        if model_name == 'resnet9':
            # your block...
            return prenet, net

        elif model_name == 'resnetwide28x10':
            # your block...
            return prenet, net

        elif model_name == 'vitb16':
            # your block...
            return prenet, net

        else:
            print(f"[divide_model] Fallback: no split for {model_name}.")
            prenet = None
            net = model
            return prenet, net

    def get_save_prefix(self):
        return

class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.cuda().eval()

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        with torch.no_grad():
            logit_t = self.og_model(images)

        loss = F.cross_entropy(output, target)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)

        if self.maximize:
            loss = -loss

        self.top1(output, target)
        return loss

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize=False
        while self.curr_step < self.opt.msteps:
            self.maximize=True
            time_start = time.process_time()
            self.train_one_epoch(loader=forget_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval(loader=test_loader)

        self.maximize=False
        while self.curr_step < self.opt.unlearn_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval(loader=test_loader)

        self.store_final_acc()  # works here too!
        return
