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
from torch.autograd import Variable
from datetime import datetime
import shutil
import json
from transforms import normalize
from fire import Fire

cuda_is_available = torch.cuda.is_available()

def cuda(x):
    return x.cuda() if cuda_is_available else x

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x.cuda(async=True), volatile=volatile))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validation(model, criterion, valid_loader):
    model.eval()
    losses = []
    accuracies = []
    batches_count = 0
    for inputs, targets in tqdm.tqdm(valid_loader):
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        acc = accuracy(outputs, targets)[0].data[0]
        accuracies.append(acc)
        batches_count += 1
        if batches_count > 400:
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
          save_predictions=None,
          n_epochs=None,
          patience=2):
    lr = args['lr']
    n_epochs = n_epochs or args['n_epochs']
    optimizer = init_optimizer(lr)

    root = Path(args['root'])
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
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(args['epoch_size'] or
                              len(train_loader) * args['batch_size']))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        if args['epoch_size']:
            tl = islice(tl, args['epoch_size'] // args['batch_size'])
        try:
            mean_loss = 0
            for i, (inputs, targets) in tqdm.tqdm(enumerate(tl)):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                targets = targets.type(torch.LongTensor).cuda()
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
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
            valid_metrics = validation(model, criterion, valid_loader)
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
    return

class Model(object):
    def train(self, architecture, fold, lr, batch_size, epochs, epoch_size):
        print("Start training with following params:",
              f"architecture = {architecture}",
              f"fold = {fold}",
              f"lr = {lr}",
              f"batch_size = {batch_size}",
              f"epochs = {epochs}",
              f"epoch_size = {epoch_size}")
        transformation = normalize()

        train_loader, valid_loader, num_classes = get_loaders(batch_size,
                                                              train_transform=transformation,
                                                              valid_transform=transformation,
                                                              n_fold=fold)
        model = self._get_model(num_classes, architecture)
        criterion = CrossEntropyLoss()

        train_kwargs = dict(
            args=dict(lr=lr, n_epochs=epochs, root="../", fold=fold, batch_size=batch_size, epoch_size=epoch_size),
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=validation,
            patience=4,
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