from torchvision.models import resnet152, resnet50, resnet101, densenet121, densenet161, densenet169, densenet201
from dataloader import get_loaders
from pathlib import Path
import random
import torch.nn as nn
import tqdm
from itertools import islice
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import torch
from datetime import datetime
import shutil
import json
from transforms import normalize
from pt_util import variable, long_tensor
from metrics import accuracy
from fire import Fire


def validation(model, criterion, valid_loader, validation_size, batch_size):
    model.eval()
    losses = []
    accuracies = []
    batches_count = validation_size // batch_size
    for i, (inputs, targets) in tqdm.tqdm(enumerate(valid_loader), total=batches_count):
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        acc = accuracy(outputs, targets)[0].data[0]
        accuracies.append(acc)
        if i > batches_count:
            break
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


def train(args,
          model: nn.Module,
          criterion, *, train_loader,
          valid_loader,
          validation,
          init_optimizer,
          architecture,
          validation_size,
          save_predictions=None,
          n_epochs=None,
          patience=2):
    lr = args['lr']
    n_epochs = n_epochs or args['n_epochs']
    optimizer = init_optimizer(lr)

    root = Path(args['root'])
    if not root.exists():
        root.mkdir()
    model_path = root / 'model_{}.pt'.format(args['fold'])
    best_model_path = root / 'best-model_{}.pt'.format(args['fold'])
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20
    log = root.joinpath('train_{}.log'.format(args['fold'])).open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    batch_size = args['batch_size']
    for epoch in range(epoch, n_epochs + 1):
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
                outputs = model(inputs)
                targets = long_tensor(targets)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, validation_size, batch_size)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr /= 5
                lr_reset_epoch = epoch
                optimizer = init_optimizer(lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            break
    return

class Model(object):
    def train(self, architecture, fold, lr, batch_size, epochs, epoch_size, validation_size):
        print("Start training with following params:",
              f"architecture = {architecture}",
              f"fold = {fold}",
              f"lr = {lr}",
              f"batch_size = {batch_size}",
              f"epochs = {epochs}",
              f"epoch_size = {epoch_size}",
              f"validation_size = {validation_size}")
        transformation = normalize()

        train_loader, valid_loader, num_classes = get_loaders(batch_size,
                                                              train_transform=transformation,
                                                              valid_transform=transformation,
                                                              n_fold=fold)
        model = self._get_model(num_classes, architecture)
        criterion = CrossEntropyLoss()

        train_kwargs = dict(
            args=dict(lr=lr, n_epochs=epochs, root=f"../results/{architecture}", fold=fold, batch_size=batch_size, epoch_size=epoch_size),
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=validation,
            validation_size=validation_size,
            patience=4,
            architecture=architecture
        )
        init_optimizer = lambda x: Adam(model.parameters(), lr=x)
        train(init_optimizer=init_optimizer, **train_kwargs)

    @staticmethod
    def _get_model(num_classes, architecture='resnet50'):
        if "resnet" in architecture:
            if architecture == 'resnet50':
                model = resnet50(pretrained=True).cuda()
            elif architecture == 'resnet101':
                model = resnet101(pretrained=True).cuda()
            elif architecture == 'resnet152':
                model = resnet152(pretrained=True).cuda()
            else:
                raise Exception(f'Unknown architecture: {architecture}')
            model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
        elif "densenet" in architecture:
            if architecture == 'densenet121':
                model = densenet121(pretrained=True).cuda()
            elif architecture == "densenet161":
                model = densenet161(pretrained=True).cuda()
            elif architecture == "densenet169":
                model = densenet169(pretrained=True).cuda()
            elif architecture == "densenet201":
                model = densenet201(pretrained=True).cuda()
            else:
                raise Exception(f'Unknown architecture: {architecture}')
            model.classifier = nn.Linear(model.classifier.in_features, num_classes).cuda()
        else:
            raise Exception(f'Unknown architectire: {architecture}')
        return nn.DataParallel(model).cuda()


if __name__ == '__main__':
    Fire(Model)