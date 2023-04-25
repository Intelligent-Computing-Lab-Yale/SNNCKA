import argparse
import shutil
import os
import time
import torch
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from models.ann_resnet import resnet20, resnet38, resnet56, resnet110, resnet164
import data_loaders
from functions import TET_loss, seed_all, mix_loss

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dset', default='c10', type=str, metavar='N', choices=['c10', 'c100'],
                    help='dataset')
parser.add_argument('--model', default='res20', type=str, metavar='N', choices=['res20', 'res38', 'res56', 'res110',
                                                                                'res164'],
                    help='neural network architecture')
parser.add_argument('--w', default=1, type=int,
                    help='width multiplier for networks. ')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed', default=1000, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--amp', action='store_true',
                    help='if use amp training.')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    s_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()

        running_loss += loss.item()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    e_time = time.time()
    return running_loss / i, 100 * correct / total, (e_time-s_time)/60


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    correct = torch.tensor([correct]).cuda()
    total = torch.tensor([total]).cuda()
    final_acc = 100 * correct / total
    return final_acc.item()


if __name__ == '__main__':

    seed_all(args.seed)

    if args.dset == 'c10':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=True)
        num_cls = 10
        wd = 1e-4
    elif args.dset == 'c100':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=False)
        num_cls = 100
        wd = 1e-4
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    if args.model == 'res20':
        model = resnet20(width_mult=args.w, in_c=3, num_classes=num_cls)
    elif args.model == 'res38':
        model = resnet38(width_mult=args.w, in_c=3, num_classes=num_cls)
    elif args.model == 'res56':
        model = resnet56(width_mult=args.w, in_c=3, num_classes=num_cls)
    elif args.model == 'res110':
        model = resnet110(width_mult=args.w, in_c=3, num_classes=num_cls)
    elif args.model == 'res164':
        model = resnet164(width_mult=args.w, in_c=3, num_classes=num_cls)
    else:
        raise NotImplementedError

    model.T = args.time
    model.cuda()
    device = next(model.parameters()).device

    scaler = GradScaler() if args.amp else None

    model_save_name = 'raw/{}_{}_ann'.format(args.dset, args.model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/256 * args.batch_size, weight_decay=wd, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    best_acc = 0
    best_epoch = 0
    print('start training!')
    for epoch in range(args.epochs):

        loss, acc, t_diff = train(model, device, train_loader, criterion, optimizer, epoch, scaler, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f},\t time elapsed: {}'.format(epoch, args.epochs, loss, acc,
                                                                                    t_diff))
        scheduler.step()
        facc = test(model, test_loader, device)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_name)
        print('Best Test acc={:.3f}'.format(best_acc))
