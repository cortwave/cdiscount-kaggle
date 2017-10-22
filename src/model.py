from dataloader import get_loaders, get_test_loader, get_valid_loader
from pathlib import Path
import random
import torch.nn as nn
import tqdm
from itertools import islice
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
import numpy as np
import torch
from datetime import datetime
import shutil
import json
from transforms import train_augm, valid_augm
from pt_util import variable, long_tensor
from metrics import accuracy
from fire import Fire
import pandas as pd
from model_factory import get_model


def validate(model, criterion, valid_loader, validation_size, batch_size, iter_size):
    model.eval()
    losses = []
    accuracies = []
    batches_count = validation_size // batch_size
    valid_loader = islice(valid_loader, batches_count)
    for i, (inputs, targets) in tqdm.tqdm(enumerate(valid_loader), total=batches_count, desc="validation"):
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        targets = long_tensor(targets)
        inputs_chunks = inputs.chunk(iter_size)
        targets_chunks = targets.chunk(iter_size)
        loss = 0
        acc = 0
        for input, target in zip(inputs_chunks, targets_chunks):
            outputs = model(input)
            loss_iter = criterion(outputs, target)
            loss_iter /= batch_size
            loss += loss_iter.data[0]
            acc_iter = accuracy(outputs, target)[0]
            acc_iter /= iter_size
            acc += acc_iter.data[0]
        losses.append(loss)
        accuracies.append(acc)
    valid_loss = np.mean(losses)
    valid_acc = np.mean(accuracies)
    print('Valid loss: {:.4f}, acc: {:.4f}'.format(valid_loss, valid_acc))
    return {'valid_loss': valid_loss, 'valid_acc': valid_acc}


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


class Model(object):
    def train(self, architecture, fold, lr, batch_size, epochs, epoch_size, validation_size, iter_size, patience=4, optim="adam"):
        print("Start training with following params:",
              f"architecture = {architecture}",
              f"fold = {fold}",
              f"lr = {lr}",
              f"batch_size = {batch_size}",
              f"epochs = {epochs}",
              f"epoch_size = {epoch_size}",
              f"validation_size = {validation_size}",
              f"iter_size = {iter_size}",
              f"optim = {optim}",
              f"patience = {patience}")

        train_loader, valid_loader, num_classes = get_loaders(batch_size,
                                                              train_transform=train_augm(),
                                                              valid_transform=valid_augm(),
                                                              n_fold=fold)
        model = get_model(num_classes, architecture)
        criterion = CrossEntropyLoss(size_average=False)

        self.lr = lr
        self.model = model
        self.root = Path(f"../results/{architecture}")
        self.fold = fold
        self.optim = optim
        train_kwargs = dict(
            args=dict(iter_size=iter_size, n_epochs=epochs,
                      batch_size=batch_size, epoch_size=epoch_size),
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation_size=validation_size,
            patience=patience
        )
        self._train(**train_kwargs)

    def _init_optimizer(self):
        if self.optim == "adam":
            return Adam(self.model.parameters(), lr=self.lr, )
        elif self.optim == "sgd":
            return SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise Exception(f"Unknown optimizer {self.optim}")

    def _init_files(self):
        if not self.root.exists():
            self.root.mkdir()
        self.log = self.root.joinpath('train_{}.log'.format(self.fold)).open('at', encoding='utf8')
        self.model_path = self.root / 'model_{}.pt'.format(self.fold)
        self.best_model_path = self.root / 'best-model_{}.pt'.format(self.fold)

    def _init_model(self):
        if self.model_path.exists():
            state = torch.load(str(self.model_path))
            self.epoch = state['epoch']
            self.step = state['step']
            self.best_valid_loss = state['best_valid_loss']
            self.model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {:,}'.format(self.epoch, self.step))
        else:
            self.epoch = 1
            self.step = 0
            self.best_valid_loss = float('inf')

    def _save_model(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch,
            'step': self.step,
            'best_valid_loss': self.best_valid_loss
        }, str(self.model_path))

    def _train(self,
               args,
               model: nn.Module,
               criterion,
               *,
               train_loader,
               valid_loader,
               validation_size,
               patience=2):
        lr = self.lr
        n_epochs = args['n_epochs']
        optimizer = self._init_optimizer()
        self._init_files()
        self._init_model()

        report_each = 10
        valid_losses = []
        lr_reset_epoch = self.epoch
        batch_size = args['batch_size']
        iter_size = args['iter_size']
        for epoch in range(self.epoch, n_epochs + 1):
            model.train()
            random.seed()
            tq = tqdm.tqdm(total=(args['epoch_size'] or
                                  len(train_loader) * batch_size))
            tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
            losses = []
            tl = train_loader
            if args['epoch_size']:
                tl = islice(tl, args['epoch_size'] // batch_size)
            try:
                mean_loss = 0
                for i, (inputs, targets) in enumerate(tl):
                    inputs, targets = variable(inputs), variable(targets)
                    targets = long_tensor(targets)
                    inputs_chunks = inputs.chunk(iter_size)
                    targets_chunks = targets.chunk(iter_size)
                    optimizer.zero_grad()

                    iter_loss = 0
                    for input, target in zip(inputs_chunks, targets_chunks):
                        outputs = model(input)
                        loss = criterion(outputs, target)
                        loss /= batch_size
                        iter_loss += loss.data[0]
                        loss.backward()
                    optimizer.step()
                    self.step += 1
                    tq.update(batch_size)
                    losses.append(iter_loss)
                    mean_loss = np.mean(losses[-report_each:])
                    tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                    if i and i % report_each == 0:
                        write_event(self.log, self.step, loss=mean_loss)
                write_event(self.log, self.step, loss=mean_loss)
                tq.close()
                self._save_model(epoch + 1)
                valid_metrics = validate(model, criterion, valid_loader, validation_size, batch_size, iter_size)
                write_event(self.log, self.step, **valid_metrics)
                valid_loss = valid_metrics['valid_loss']
                valid_losses.append(valid_loss)
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    shutil.copy(str(self.model_path), str(self.best_model_path))
                elif (patience and epoch - lr_reset_epoch > patience and
                              min(valid_losses[-patience:]) > self.best_valid_loss):
                    lr /= 1.1
                    lr_reset_epoch = epoch
                    optimizer = self._init_optimizer()
            except KeyboardInterrupt:
                tq.close()
                print('Ctrl+C, saving snapshot')
                self._save_model(epoch)
                print('done.')
                break
        return

    def predict(self, architecture, fold, tta, batch_size, name="sub"):
        print("Start predicting with following params:",
              f"architecture = {architecture}",
              f"fold = {fold}",
              f"tta = {tta}")
        n_classes = 5270
        model = get_model(num_classes=n_classes, architecture=architecture)
        state = torch.load(f"../results/{architecture}/best-model_{fold}.pt")
        model.load_state_dict(state['model'])
        test_augm = valid_augm()
        label_map = pd.read_csv("../data/labels_map.csv")
        label_map.index = label_map['label_id']
        test_loader = get_test_loader(batch_size, test_augm)
        with open(f"../results/{architecture}/{name}_{fold}.csv", "w") as f:
            f.write("_id,category_id\n")
            for images, product_ids in tqdm.tqdm(test_loader):
                images = variable(images)
                preds = model(images).data.cpu().numpy()
                for pred, product_id in zip(preds, product_ids):
                    label = np.argmax(pred, 0)
                    cat_id = label_map.ix[label]['category_id']
                    f.write(f"{product_id},{cat_id}\n")

    def predict_validation(self, architecture, fold, tta, batch_size):
        n_classes = 5270
        model = get_model(num_classes=n_classes, architecture=architecture)
        state = torch.load(f"../results/{architecture}/best-model_{fold}.pt")
        model.load_state_dict(state['model'])
        test_augm = valid_augm()
        label_map = pd.read_csv("../data/labels_map.csv")
        label_map.index = label_map['label_id']
        loader = get_valid_loader(fold, batch_size, test_augm)
        with open(f"../results/{architecture}/validation_{fold}.csv", "w") as f:
            f.write("_id,category_id\n")
            for images, product_ids in tqdm.tqdm(loader):
                images = variable(images)
                preds = model(images).data.cpu().numpy()
                for pred, product_id in zip(preds, product_ids):
                    label = np.argmax(pred, 0)
                    cat_id = label_map.ix[label]['category_id']
                    f.write(f"{product_id},{cat_id}\n")


if __name__ == '__main__':
    Fire(Model)
