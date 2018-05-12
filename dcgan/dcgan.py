from __future__ import print_function
import argparse
import os
from torch.autograd import Variable
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch

parser = argparse.ArgumentParser(description="DCGAN Pytorch")
parser.add_argument('--workers', default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imagesize', type=int, default=64, help='the height/width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoint')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

print("Random seed: ", args.manualSeed)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# folder dataset
dataset = dsets.ImageFolder(root='data/',
                            transform = transforms.Compose([
                                transforms.Resize(args.imagesize),
                                transforms.CenterCrop(args.imagesize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True,
                                         num_workers=args.workers, drop_last=True)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=100, out_channels=64*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.ReLU(True),
            # state size. (64*8)*4*4
            nn.ConvTranspose2d(in_channels=64*8, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.ReLU(True),
            # state size. (64*4)*8*8
            nn.ConvTranspose2d(in_channels=64*4, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.ReLU(True),
            # state size. (64*2)*16*16
            nn.ConvTranspose2d(in_channels=64*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # state size. (64)*32*32
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.Tanh(),
            # state size. (3)*64*64
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator(ngpu=args.ngpu)
netG.apply(weights_init)

if args.netG != '':
    netG.load_state_dict(torch.load(netG))
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 64 * 32 * 32
            nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64 * 2) * 16 * 16
            nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64*4) * 8 * 8
            nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64*8) * 4 * 4
            nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu=args.ngpu)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

criterion = nn.BCELoss()
fixed_noise = Variable(torch.randn(args.batchSize, 100, 1, 1)).cuda()
real_label = 1
fake_label = 0
if cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()

# setup optimizer
optimizerD = optim.Adam(params=netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(params=netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


for epoch in range(args.niter):
    for i, inputs in enumerate(dataloader, 0):
        # update D network: maximize log(D(x) + log(1-D(G(z)))
        # train with real
        netD.zero_grad()
        real_cpu = Variable(inputs[0]).cuda()
        cpu_data = inputs[0]
        batch_size = inputs[0].size(0)
        label = Variable(torch.ones(args.batchSize)).cuda()
        output = netD(real_cpu)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().data[0]

        # train with fake
        noise = Variable(torch.randn(args.batchSize, 100, 1, 1)).cuda()
        # generate the fake picture
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().data[0]
        errD = errD_real + errD_fake
        optimizerD.step()

        # update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label) # fake label are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().data[0]
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(cpu_data,
                              '%s/result/real_samples.png' % args.outf,
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach().data,
                              '%s/result/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                              normalize=True)

            # do checkpointing
    torch.save(netG.state_dict(), '%s/checkpoint/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/checkpoint/netD_epoch_%d.pth' % (args.outf, epoch))





