#!/usr/bin/env python3

import argparse
import itertools
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os

from models import Generator, Discriminator
from datasets import LunarLanderDataset, DummyVectorDataset

# Custom LambdaLR for learning rate scheduling
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# Replay buffer to store past generated samples
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if np.random.uniform(0,1) > 0.5:
                    i = np.random.randint(0, self.max_size)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)

# Initialize network weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Simple logger for metrics
class Logger:
    def __init__(self, n_epochs, batches_per_epoch):
        self.n_epochs = n_epochs
        self.batches_per_epoch = batches_per_epoch
        self.epoch = 0
        self.batch = 0
        self.metrics = {}
        
    def log(self, losses=None, metrics=None):
        if losses:
            for k, v in losses.items():
                if k not in self.metrics:
                    self.metrics[k] = []
                self.metrics[k].append(v.item())
                
        self.batch += 1
        if self.batch >= self.batches_per_epoch:
            self.batch = 0
            self.epoch += 1
            
            # Print progress
            print(f"Epoch {self.epoch}/{self.n_epochs}")
            if self.metrics:
                for k, v in self.metrics.items():
                    avg = sum(v[-self.batches_per_epoch:]) / self.batches_per_epoch
                    print(f"  {k}: {avg:.5f}")
                    
                    # Save metrics periodically
                    if not os.path.exists('./output'):
                        os.makedirs('./output')
                    np.save(f'./output/{k}.npy', np.array(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./raw_data/', help='root directory of the raw data')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=150, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--input_dim', type=int, default=900, help='dimension of input data vectors')
    parser.add_argument('--hidden_dim', type=int, default=1800, help='dimension of hidden layers in generator')
    parser.add_argument('--n_residual', type=int, default=5, help='number of residual blocks in generator')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--use_dummy', action='store_true', help='use dummy data for testing')
    parser.add_argument('--unaligned', action='store_true', default=True, help='use unaligned data (random pairing)')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device('mps' if opt.cuda else 'cpu')

    # Networks
    netG_A2B = Generator(opt.input_dim, opt.input_dim, opt.hidden_dim, opt.n_residual, opt.dropout).to(device)
    netG_B2A = Generator(opt.input_dim, opt.input_dim, opt.hidden_dim, opt.n_residual, opt.dropout).to(device)
    netD_A = Discriminator(opt.input_dim, opt.dropout).to(device)
    netD_B = Discriminator(opt.input_dim, opt.dropout).to(device)

    # Initialize weights
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # LR schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loading
    if opt.use_dummy:
        dataset = DummyVectorDataset(size=1000, dimension=opt.input_dim)
    else:
        dataset = LunarLanderDataset(opt.dataroot, mode='train', unaligned=opt.unaligned)
        
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Logger
    logger = Logger(opt.n_epochs, len(dataloader))

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Training
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1), requires_grad=False).to(device)

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, valid)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, valid)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, valid)

            # Fake loss
            fake_A_buffer_data = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake = netD_A(fake_A_buffer_data)
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            fake_B_buffer_data = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake = netD_B(fake_B_buffer_data)
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            # Log losses
            logger.log({
                'loss_G': loss_G,
                'loss_G_identity': (loss_identity_A + loss_identity_B),
                'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                'loss_D': (loss_D_A + loss_D_B)
            })

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')

if __name__ == '__main__':
    main()