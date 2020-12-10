import sys
import argparse
import os
import time
import copy
import random
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parseArg():
    parser = argparse.ArgumentParser(description='cDCGAN a generative adversarial model')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for cDCGAN')

    conf_file = os.path.abspath("../conf/mnist_dpcDCGAN.json")

    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


conf_file = parseArg()

# 从conf中读进超参数
with open(conf_file, "r") as f:
    conf_dict = eval(f.read())
    result_dir = conf_dict['result_dir']
    checkpoint_dir = conf_dict['checkpoint_dir']
    images_dir = conf_dict['images_dir']
    batch_size = conf_dict['batch_size']
    virtual_batch_size = conf_dict['virtual_batch_size']
    cond_num = conf_dict['cond_num']
    z_dim = conf_dict['hidden_dim']
    pic_dim = conf_dict['pic_dim']
    train_epoch = conf_dict['epochs']
    img_size = conf_dict['img_size']
    is_train = conf_dict['is_train']
    is_test = conf_dict['is_test']
    is_load = conf_dict['is_load']
    penalty_factor = conf_dict['penalty_factor']
    percentage = conf_dict['percentage']
    input_dataset = conf_dict['dataset']
    learning_rate_initial = conf_dict['learning_rate_initial']
    decay_steps = conf_dict['decay_steps']
    learning_rate_end = conf_dict['learning_rate_end']
    optimizer_type = conf_dict['optimizer_type']
    micro_batches = conf_dict['nr_microbatch']  # 每个batch的数据还会被划分进更小的microbacth中
    norm_clip = conf_dict['norm_clip']  # Does NOT affect EPSILON, but increases NOISE on gradients
    noise_mult = conf_dict['noise_mult']
    dp_delta = conf_dict['dp_delta']  # Needs to be smaller than 1/BUFFER_SIZE
    n_disc = conf_dict['disc_train_step']  # Number of times we train DISC before training GEN once
    generate_num = conf_dict['pic_gen_num']

del conf_dict, conf_file

assert virtual_batch_size % batch_size == 0
virtual_batch_rate = int(virtual_batch_size / batch_size)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
if not os.path.exists(result_dir + images_dir):
    os.mkdir(result_dir + images_dir)
if not os.path.exists(result_dir + checkpoint_dir):
    os.mkdir(result_dir + checkpoint_dir)

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
if input_dataset == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
elif input_dataset == "FASHION_MNIST":
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

template_z = torch.randn(cond_num, z_dim)
template_y = torch.zeros(cond_num, 1)
test_z = template_z
for i in range(cond_num - 1):
    test_z = torch.cat([test_z, template_z], 0)
    temp = torch.ones(cond_num, 1) + i
    template_y = torch.cat([template_y, temp], 0)
test_z = test_z.view(-1, z_dim, 1, 1)
test_label = torch.zeros(z_dim, cond_num)
test_label.scatter_(1, template_y.type(torch.LongTensor), 1)
test_label = test_label.view(-1, cond_num, 1, 1)
test_z, test_label = Variable(test_z.to(device)), Variable(test_label.to(device))
# 训练标签模版
one_hot = torch.zeros(cond_num, cond_num)
one_hot = one_hot.scatter_(1, torch.LongTensor(range(cond_num)).view(cond_num, 1), 1).view(cond_num, cond_num, 1, 1)
fill = torch.zeros([cond_num, cond_num, img_size, img_size])
for i in range(cond_num):
    fill[i, i, :, :] = 1


class Generator(nn.Module):
    def __init__(self, z_dim, cond_num, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim + cond_num, d * 4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, pic_dim, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, pic_dim, cond_dim, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(pic_dim + cond_dim, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def save_result(num_epoch, path='result.png'):
    with torch.no_grad():
        test_images = generator(test_z, test_label)

    size_figure_grid = cond_num
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(cond_num * cond_num):
        i = k // cond_num
        j = k % cond_num
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


def save_train_hist(hist, path='Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)


def compute_gradient_penalty(model, real, fake, real_label, fake_label):
    alpha = torch.Tensor(np.random.random((real.size(0), 1, img_size, img_size))).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).to(device).requires_grad_(True)
    interpolates.retain_grad()
    interpolates_label = (alpha * real_label + (1 - alpha) * fake_label).to(device).requires_grad_(True)
    D_interpolates = model(interpolates, interpolates_label).view(-1, 1)
    fake_out = Variable(torch.Tensor(real.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    D_interpolates.backward(fake_out, retain_graph=True)
    gradients_pic = interpolates.grad.view(real.size(0), -1).requires_grad_(True)
    gradients_label = interpolates_label.grad.view(real.size(0), -1).requires_grad_(True)
    gradient_penalty = ((gradients_pic.norm(2, dim=1) - 1) ** 2).mean() + (
                (gradients_label.norm(2, dim=1) - 1) ** 2).mean()
    discriminator.zero_grad()
    return gradient_penalty


generator = Generator(z_dim, cond_num)
discriminator = Discriminator(pic_dim, cond_num)
generator = convert_batchnorm_modules(generator)
discriminator = convert_batchnorm_modules(discriminator)
generator.to(device)
discriminator.to(device)
generator.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

loss_function = nn.BCELoss()

if optimizer_type == 'Adam':
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate_initial, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate_initial, betas=(0.5, 0.999))
elif optimizer_type == 'GD':
    G_optimizer = optim.SGD(generator.parameters(), lr=learning_rate_initial)
    D_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate_initial)
privacy_engine = PrivacyEngine(discriminator,
                               batch_size=virtual_batch_size,
                               sample_size=len(train_loader.dataset),
                               alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                               noise_multiplier=noise_mult,
                               max_grad_norm=norm_clip)
privacy_engine.attach(D_optimizer)
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_times'] = []

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()

    for idx, (feature, label) in enumerate(train_loader):
        batch_size = feature.size()[0]
        '''训练判别器'''
        D_total_loss = 0
        for step in range(n_disc):
            # 评价所用标签
            y_real = torch.ones(batch_size)
            y_fake = torch.zeros(batch_size)
            y_real, y_fake = Variable(y_real.to(device)), Variable(y_fake.to(device))
            real_fill = fill[label]
            feature, real_fill = Variable(feature.to(device)), Variable(real_fill.to(device))
            # 用真实图像训练
            # print('DIS_REAL ',idx)
            D_result = discriminator(feature, real_fill).squeeze()
            D_real_loss = loss_function(D_result, y_real)
            discriminator.zero_grad()
            D_real_loss.backward()
            D_optimizer.step()
            # 生成所用标签
            # print('DIS_FAKE ',idx)
            gen_z = torch.randn((batch_size, z_dim)).view(-1, z_dim, 1, 1)
            label = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
            gen_label = one_hot[label]
            gen_fill = fill[label]
            gen_z, gen_label, gen_fill = Variable(gen_z.to(device)), Variable(gen_label.to(device)), Variable(
                gen_fill.to(device))
            # 用生成图像训练
            G_result = generator(gen_z, gen_label)
            D_result = discriminator(G_result.detach(), gen_fill).squeeze()
            D_fake_loss = loss_function(D_result, y_fake)
            # 计算损失
            discriminator.zero_grad()
            D_fake_loss.backward()
            D_optimizer.step()

            D_penalty_loss = penalty_factor * compute_gradient_penalty(discriminator, feature, G_result, real_fill,
                                                                       gen_fill)
            discriminator.zero_grad()
            D_penalty_loss.backward()
            D_optimizer.step()
            D_train_loss = D_real_loss + D_fake_loss + D_penalty_loss
            D_total_loss += D_train_loss.data.item()
        D_losses.append(D_total_loss / n_disc)

        '''训练生成器'''
        # print('GEN ',idx)
        generator.zero_grad()
        # 生成所用标签
        gen_z = torch.randn((batch_size, z_dim)).view(-1, z_dim, 1, 1)
        label_gen = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).squeeze()
        gen_label = one_hot[label_gen]
        gen_fill = fill[label_gen]
        gen_z, gen_label, gen_fill = Variable(gen_z.to(device)), Variable(gen_label.to(device)), Variable(
            gen_fill.to(device))
        # 用生成图像训练
        G_result = generator(gen_z, gen_label)
        # 否则会对原discriminator计算造成干扰
        discriminator_copy = copy.deepcopy(discriminator)
        D_result = discriminator_copy(G_result, gen_fill).squeeze()
        # 计算损失
        G_train_loss = loss_function(D_result, y_real)
        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.item())
        del discriminator_copy

    epoch_end_time = time.time()
    per_epoch_time = epoch_end_time - epoch_start_time
    print('Epoch %d - time: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), per_epoch_time, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
    epsilon, best_alpha = D_optimizer.privacy_engine.get_privacy_spent(dp_delta)
    print(f"(ε = {epsilon:.2f}, δ = {dp_delta}) for α = {best_alpha}")
    pic_path = result_dir + images_dir + '/' + str(epoch + 1) + '.png'
    save_result((epoch + 1), path=pic_path)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_times'].append(per_epoch_time)

print("Saving Models")
torch.save(generator.state_dict(), result_dir + checkpoint_dir + '/generator_param.pkl')
torch.save(discriminator.state_dict(), result_dir + checkpoint_dir + '/discriminator_param.pkl')
with open(result_dir + checkpoint_dir + '/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
save_train_hist(train_hist, path=result_dir + images_dir + '/train_hist.png')
