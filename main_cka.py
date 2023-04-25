import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import models.ann_resnet as ann_res
import models.snn_resnet as snn_res
import models.layers as layers
import data_loaders
from functions import seed_all

# for original CKA computation


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--dset', default='c10', type=str, metavar='N', choices=['c10', 'c100'],
                    help='dataset')
parser.add_argument('-b', '--batch_size', default=1024, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training. ')
parser.add_argument('--repeat', default=2, type=int, help='repeat times for computing CKA. ')
args = parser.parse_args()


class CKA:
    def __init__(self, model1: nn.Module, model2: nn.Module, loader, device='cuda'):
        self.model1 = model1
        self.model1.eval()
        self.model2 = model2
        self.model2.eval()
        self.device = device
        self.init_var()

    def init_var(self):
        self.Ks = []
        self.Ls = []
        self.hsic_k = 0
        self.hsic_l = 0
        self.hsic_kl = 0

    def hook_layer(self, spiking1=False, spiking2=False):
        n = args.batch_size
        mask = torch.eye(n, n).bool().to(device)

        def get_gram_k(layer, inputs, outputs):
            if not layer.training:
                # o_m = outputs.mean(1)
                gram = torch.matmul(outputs.flatten(1), outputs.flatten(1).T)
                self.Ks += [gram.masked_fill_(mask, 0)]

        def get_gram_l(layer, inputs, outputs):
            if not layer.training:
                gram = torch.matmul(outputs.flatten(1), outputs.flatten(1).T)
                self.Ls += [gram.masked_fill_(mask, 0)]

        def is_layer(m, spiking, name=None):
            if spiking:
                if isinstance(m, (layers.LIFSpike, snn_res.PreActBasicBlock)):
                    return True
                elif isinstance(m, layers.SeqToANNContainer):
                    return not isinstance(m.module, nn.Linear)
                elif name == 'conv1':
                    return True
                else:
                    return False
            else:
                return isinstance(m, (nn.Conv2d, nn.ReLU, nn.BatchNorm2d, ann_res.PreActBasicBlock,
                                      nn.AdaptiveAvgPool2d, nn.AvgPool2d))

        for name, module in self.model1.named_modules():
            if is_layer(module, spiking1, name):
                module.register_forward_hook(get_gram_k)

        for name, module in self.model2.named_modules():
            if is_layer(module, spiking2, name):
                module.register_forward_hook(get_gram_l)

    @torch.no_grad()
    def inference(self, loader, single_mode=False, attack=False, **kwargs):
        iter_loader = iter(loader)
        for i in range(args.repeat):
            data, label = next(iter_loader)
            data = data.cuda()
            data2 = copy.deepcopy(data)

            if attack:
                self.model2.train()
                data2 = pgd_attack(self.model2, data2, label, self.device, nn.CrossEntropyLoss(), **kwargs)
                self.model2.eval()

            self.model1(data)
            # compute CKA
            n = data.shape[0]
            ones = torch.ones((n, 1)).to(self.device)
            hsic_k = [self.get_hsic(K, K, n, ones) for K in self.Ks]

            if single_mode:
                hsic_l = copy.deepcopy(hsic_k)
                hsic_kl = [[self.get_hsic(K, L, n, ones) for K in self.Ks] for L in self.Ks]
            else:
                self.model2(data2)
                hsic_l = [self.get_hsic(L, L, n, ones) for L in self.Ls]
                hsic_kl = [[self.get_hsic(K, L, n, ones) for K in self.Ks] for L in self.Ls]

            # dist.barrier()
            self.Ks = []
            self.Ls = []
            torch.cuda.empty_cache()

            hsic_k = torch.stack(hsic_k)
            hsic_l = torch.stack(hsic_l)
            hsic_kl = torch.stack([torch.stack(t) for t in hsic_kl])

            self.hsic_k = self.hsic_k + hsic_k
            self.hsic_l = self.hsic_l + hsic_l
            self.hsic_kl = self.hsic_kl + hsic_kl

        self.hsic_k = torch.sqrt(self.hsic_k)
        self.hsic_l = torch.sqrt(self.hsic_l)
        self.hsic_kl = self.hsic_kl
        l_k = self.hsic_k.numel()
        l_l = self.hsic_l.numel()
        hsic = self.hsic_kl.squeeze() / (self.hsic_l.reshape(l_l, 1) @ self.hsic_k.reshape(1, l_k))
        return hsic

    def get_hsic(self, K, L, n, ones):

        return 1 / n / (n - 3) * (torch.trace(K @ L) + (ones.T @ K @ ones @ ones.T @ L @ ones) / (n - 1) / (n - 2) -
                                  2 * (ones.T @ K @ L @ ones) / (n - 2))


@torch.enable_grad()
def pgd_attack(model, data, label, device, criterion, eps=0.01, iters=20):
    corrupt_data = torch.clone(data).requires_grad_()
    label = label.to(device)
    alpha = eps / 10
    for i in range(iters):
        duplicates = torch.clone(corrupt_data)
        outputs = model(duplicates)
        if len(outputs.shape) == 3:
            outputs = outputs.mean(1)
        loss = criterion(outputs, label)
        loss.backward()

        corrupt_data.data += alpha * corrupt_data.grad.sign()
        eta = torch.clamp(corrupt_data.data - data.data, min=-eps, max=eps)
        corrupt_data.data = data.data + eta
        corrupt_data.grad.zero_()

    corrupt_data.requires_grad = False
    return corrupt_data


@torch.no_grad()
def test(model, test_loader, device, attack=False, **kwargs):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)

        if attack:
            inputs = pgd_attack(model, inputs, targets, device, nn.CrossEntropyLoss(), **kwargs)

        outputs = model(inputs)
        if len(outputs.shape) == 3:
            outputs = outputs.mean(1)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    correct = torch.tensor([correct]).cuda()
    total = torch.tensor([total]).cuda()
    final_acc = 100 * correct / total
    return final_acc.item()


if __name__ == '__main__':

    seed_all(args.seed)

    if args.dset == 'c10':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=True, cutout=True, auto_aug=True)
        num_cls = 10
        wd = 1e-4
        in_c = 3
    elif args.dset == 'c100':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=False, cutout=True, auto_aug=True)
        num_cls = 100
        wd = 5e-4
        in_c = 3
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    device = 'cuda'
    model1 = snn_res.resnet20(width_mult=1, in_c=3, mode='normal', num_classes=10)
    model1.load_state_dict(torch.load('raw/c10_res20_snn'))
    model1.cuda()

    model2 = ann_res.resnet20(width_mult=1, in_c=3, mode='normal', num_classes=10)
    model2.load_state_dict(torch.load('raw/c10_res20_ann'))
    model2.cuda()

    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False

    cka = CKA(model1, model2, test_loader, device)
    print('Registering hook to model layers...')
    cka.hook_layer(spiking1=True, spiking2=False)

    print('Computing centered kernel alignment...')
    hsic = cka.inference(loader=test_loader, single_mode=False)
    hsic = hsic.cpu().numpy()
    np.save('cka/asnn_res20.npy', hsic)
