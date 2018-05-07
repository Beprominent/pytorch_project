import argparse
import os
import shutil
import torch
import torch.cuda
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision.transforms import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from models import *
import time
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Pytorch cifar10")
parser.add_argument('--lr', type=int, default=0.1, help='--learning rate')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight-decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('-b', '--batch_size', default=128, type=int, help='batch size')
args = parser.parse_args()

best_prec = 0

def main():
    global best_prec
    use_gpu = torch.cuda.is_available()
    
    print('=> Building model...')
    if use_gpu:
        model = VGG('VGG11')
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/cifar10'
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        model = torch.nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    
    else:
        print('Cuda is not available!')
        return 

    
    if args.resume:
        if os.path.isfile(args.resume):
            print('=>loading checkpoint: {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    # Data loading and preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])
    train_dataset = dsets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = dsets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=2)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_dataloader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(test_dataloader, model, criterion)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)


class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # input = Variable(input).cuda()
        # target = Variable(target).cuda()
        input, target = input.cuda(), target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute the loss
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val: .3f} (batch_time.avg: .3f) \t'
                  'Data {data_time.val: .3f} ({data_time.avg: .3f}) \t'
                  'Loss {loss.val: .3f} ({loss.avg: .3f}) \t'
                  'Prec {top1.val: .3f}% ({top1.avg: .3f}%)'.format(epoch,
                    batch_idx, len(train_dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

def validate(test_dataloader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for batch_idx, (input, target) in enumerate(test_dataloader):
        # measure data loading time

        input, target = input.cuda(), target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute the loss
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec[0], input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                batch_idx, len(test_dataloader), batch_time=batch_time, loss=losses,
                top1=top1))
    print(' * Prec {top1.avg: .3f}%'.format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 in 80, 120 epochs"""
    if epoch < 80:
        lr = args.lr
    elif epoch < 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
        
        