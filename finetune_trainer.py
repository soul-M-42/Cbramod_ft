import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator
from loss import SimCLRLoss


class Trainer(object):
    def __init__(self, params, data_loader, model, writer=None):
        self.params = params
        self.data_loader = data_loader
        self.writer = writer

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.cls_criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
            self.reg_criterion = MSELoss().cuda()
            self.sim_criterion = SimCLRLoss(temperature=0.07).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': 0.001*(self.params.batch_size/256)**0.5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def _log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _log_text(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_text(tag, value, step)

    def _log_learning_rates(self, epoch, optim_state):
        for index, param_group in enumerate(optim_state['param_groups']):
            self._log_scalar('train/lr_group_{}'.format(index), param_group['lr'], epoch)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses_total = []
            losses_cls = []
            losses_reg = []
            losses_sim = []
            if(1):
                for x, y in tqdm(self.data_loader['train'], mininterval=10):
                    self.optimizer.zero_grad()
                    x = x.cuda()
                    y = y.cuda()
                    cls_label = y[:, -1].long()
                    reg_label = y[:, :-1]
                    feats, pred, reg = self.model(x)
                    if self.params.downstream_dataset == 'ISRUC':
                        loss = self.criterion(pred.transpose(1, 2), y)
                    else:
                        cls_loss = self.cls_criterion(pred, cls_label)
                        reg_loss = self.reg_criterion(reg, reg_label)
                        sim_loss, sim_logits, sim_labels, [sim_acc_1, sim_acc_5] = self.sim_criterion(feats)
                        # loss = cls_loss + reg_loss + sim_loss
                        loss = cls_loss + sim_loss * 0.1

                        # loss = cls_loss + reg_loss * 0.0

                    loss.backward()
                    losses_total.append(loss.data.cpu().numpy())
                    losses_cls.append(cls_loss.data.cpu().numpy())
                    losses_reg.append(reg_loss.data.cpu().numpy())
                    losses_sim.append(sim_loss.data.cpu().numpy())
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()
            train_loss = float(np.mean(losses_total))
            cls_loss = float(np.mean(losses_cls))
            reg_loss = float(np.mean(losses_reg))
            sim_loss = float(np.mean(losses_sim))
            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f} = cls {:.5f} + reg {:.5f} , acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        train_loss,
                        cls_loss,
                        reg_loss,
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                self._log_scalar('train/loss', train_loss, epoch + 1)
                self._log_scalar('train/cls_loss', cls_loss, epoch + 1)
                self._log_scalar('train/reg_loss', reg_loss, epoch + 1)
                self._log_scalar('val/acc', acc, epoch + 1)
                self._log_scalar('val/kappa', kappa, epoch + 1)
                self._log_scalar('val/f1', f1, epoch + 1)
                self._log_text('val/confusion_matrix', str(cm), epoch + 1)
                self._log_learning_rates(epoch + 1, optim_state)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            self._log_scalar('best/val_acc', acc_best, best_f1_epoch)
            self._log_scalar('best/val_kappa', kappa_best, best_f1_epoch)
            self._log_scalar('best/val_f1', f1_best, best_f1_epoch)
            self._log_text('best/val_confusion_matrix', str(cm_best), best_f1_epoch)
            self._log_scalar('test/acc', acc, best_f1_epoch)
            self._log_scalar('test/kappa', kappa, best_f1_epoch)
            self._log_scalar('test/f1', f1, best_f1_epoch)
            self._log_text('test/confusion_matrix', str(cm), best_f1_epoch)
            self._log_text('artifacts/model_path', model_path, best_f1_epoch)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()
            train_loss = float(np.mean(losses))

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        train_loss,
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                self._log_scalar('train/loss', train_loss, epoch + 1)
                self._log_scalar('val/acc', acc, epoch + 1)
                self._log_scalar('val/pr_auc', pr_auc, epoch + 1)
                self._log_scalar('val/roc_auc', roc_auc, epoch + 1)
                self._log_text('val/confusion_matrix', str(cm), epoch + 1)
                self._log_learning_rates(epoch + 1, optim_state)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            self._log_scalar('best/val_acc', acc_best, best_f1_epoch)
            self._log_scalar('best/val_pr_auc', pr_auc_best, best_f1_epoch)
            self._log_scalar('best/val_roc_auc', roc_auc_best, best_f1_epoch)
            self._log_text('best/val_confusion_matrix', str(cm_best), best_f1_epoch)
            self._log_scalar('test/acc', acc, best_f1_epoch)
            self._log_scalar('test/pr_auc', pr_auc, best_f1_epoch)
            self._log_scalar('test/roc_auc', roc_auc, best_f1_epoch)
            self._log_text('test/confusion_matrix', str(cm), best_f1_epoch)
            self._log_text('artifacts/model_path', model_path, best_f1_epoch)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()
            train_loss = float(np.mean(losses))

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        train_loss,
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                self._log_scalar('train/loss', train_loss, epoch + 1)
                self._log_scalar('val/corrcoef', corrcoef, epoch + 1)
                self._log_scalar('val/r2', r2, epoch + 1)
                self._log_scalar('val/rmse', rmse, epoch + 1)
                self._log_learning_rates(epoch + 1, optim_state)
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            self._log_scalar('best/val_corrcoef', corrcoef_best, best_r2_epoch)
            self._log_scalar('best/val_r2', r2_best, best_r2_epoch)
            self._log_scalar('best/val_rmse', rmse_best, best_r2_epoch)
            self._log_scalar('test/corrcoef', corrcoef, best_r2_epoch)
            self._log_scalar('test/r2', r2, best_r2_epoch)
            self._log_scalar('test/rmse', rmse, best_r2_epoch)
            self._log_text('artifacts/model_path', model_path, best_r2_epoch)