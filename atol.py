# -*- coding: utf-8 -*-
import numpy as np
import os, sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import utils.svhn_loader as svhn
import pdb

from models.wrn import WideResNet
from utils.display_results import get_measures, print_measures
from utils.losses import proxy_align_loss

parser = argparse.ArgumentParser(description='ATOL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# EG specific
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
# ATOL Training Setup
parser.add_argument('--mean',type=float,default=5, help="mean")
parser.add_argument('--std',type=float,default=0.1, help="std")
parser.add_argument("--ood_space_size", type=float, default=4, help="ood_space_size")
parser.add_argument('--trade_off',type=float,default=1)
# Generator
parser.add_argument("--size", type=int, default=32, help="output image size of the generator")
parser.add_argument("--latent", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--generator", '-g', type=str, default='dcgan', help="generator")

args = parser.parse_args()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
cudnn.benchmark = True  # fire on all cylinders


############# Dataset ##############
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR10('../../data/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('../../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('../../data/cifarpy', train=False, transform=test_transform)
    num_classes = 100

train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)

texture_data = dset.ImageFolder(root="../../data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
places365_data = dset.ImageFolder(root="../../data/places365/", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
svhn_data = svhn.SVHN(root='../../data/svhn/', split="test",transform=trn.Compose( [trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
lsunc_data = dset.ImageFolder(root="../../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunr_data = dset.ImageFolder(root="../../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data = dset.ImageFolder(root="../../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


############# Generator ##############
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

def generate_dcgan(noise, generator):
    noise = noise.cuda()
    noise = noise.unsqueeze(dim=2)
    noise = noise.unsqueeze(dim=3)
    output = generator(noise)
    output = torch._adaptive_avg_pool2d(input=output, output_size=(args.size, args.size))    
    return output

g_ema = Generator(args.latent, 64, 3).cuda()
g_ema.load_state_dict(torch.load('./ckpt/%s_%s.pt' % (args.generator, args.dataset)))

############# Classifier ##############
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
if args.dataset == 'cifar10':
    model_path = './ckpt/cifar10_wrn_pretrained_epoch_99.pt'
else:
    model_path = './ckpt/cifar100_wrn_pretrained_epoch_99.pt'
net.load_state_dict(torch.load(model_path), strict=False)


############# Alignment Loss ##############
VHL_mapping_matrix = torch.rand(128, 128)
align_loss = proxy_align_loss(
    inter_domain_mapping=False,
    inter_domain_class_match=True,
    noise_feat_detach=False,
    noise_contrastive=False,
    inter_domain_mapping_matrix=VHL_mapping_matrix,
    inter_domain_weight=0.0,
    inter_class_weight=1.0,
    noise_supcon_weight=0.1,
    noise_label_shift=num_classes,
    device="cuda"
    )


############# OOD Detection Results ##############
def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)
            smax = to_np(output)
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr


############# Optimizer ##############
optimizer = torch.optim.SGD(
         net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))


############# Parameter ##############
means = torch.ones(num_classes,args.latent) * args.mean
means = torch.where(torch.rand_like(means) > 0.5, means, -means)
auxiliary_id_bs = args.batch_size
auxiliary_ood_bs = args.batch_size * 4


############# Training ##############
def train(epoch):
    
    net.train()  # enter train mode

    loss_avg = 0
    noise = (torch.rand(auxiliary_ood_bs, args.latent) - .5) * args.mean * args.ood_space_size
    auxiliary_ood = generate_dcgan(noise, g_ema)
    for batch_idx, in_set in enumerate(train_loader_in):
        
        real_id, target = in_set[0], in_set[1]

        mog = torch.randn(auxiliary_id_bs, args.latent) * args.std
        mog += means[target]
        auxiliary_id = generate_dcgan(mog, g_ema)
        
        id_data = torch.cat((real_id.cuda(), auxiliary_id.cuda()), 0) 
        target = torch.cat((target, target), 0).cuda()
        
        id_data = id_data.detach()
        ood_data = auxiliary_ood.detach()
        data = torch.cat((id_data, ood_data), 0)

        x = net(data)
        id_feat = net.forward_feature(id_data)

        align_domain_loss_value = 0.0
        align_cls_loss_value = 0.0
        noise_cls_loss_value = 0.0
        loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = align_loss(id_feat, target, args.batch_size)
        loss_cla = -((0.5 * torch.eye(num_classes).cuda()[target] + 0.5 / num_classes) * x[:len(id_data)].log_softmax(1)).sum(-1).mean()
        loss_cla += - (x[len(id_data):].mean(1) -torch.logsumexp(x[len(id_data):], dim=1)).mean()
        loss_cla += args.trade_off * loss_feat_align

        optimizer.zero_grad()
        loss_cla.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1) 
        optimizer.step()
        scheduler.step()

        loss_avg = loss_avg * 0.8 + float(loss_cla) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_avg))
    print()


############# Testing ##############
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader.dataset)


############# Main ##############
print('Beginning Training\n')
for epoch in range(0, args.epochs + 1):
    train(epoch)
    net.eval()
    print(f"epoch: {epoch}")
    print('  FPR95 AUROC AUPR')
    in_score = get_ood_scores(test_loader, in_dist=True)
    metric_ll = []  
    print('svhn')
    metric_ll.append(get_and_print_results(svhn_loader, in_score))
    print('lsunc')
    metric_ll.append(get_and_print_results(lsunc_loader, in_score))
    print('lsunr')
    metric_ll.append(get_and_print_results(lsunr_loader, in_score))        
    print('isun')
    metric_ll.append(get_and_print_results(isun_loader, in_score))
    print('texture')
    metric_ll.append(get_and_print_results(texture_loader, in_score))
    print('places')
    metric_ll.append(get_and_print_results(places365_loader, in_score))
        
    print('total')
    print('& %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))