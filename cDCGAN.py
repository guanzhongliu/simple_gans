import sys
import argparse
import os
import time
import copy
import random
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np
from load_data import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parseArg():
    parser = argparse.ArgumentParser(description='cDCGAN a generative adversarial model')
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for cDCGAN')

    conf_file = os.path.abspath("conf/mnist_cDCGAN.json")

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
    percentage = conf_dict['percentage']
    input_dataset = conf_dict['dataset']
    learning_rate_initial = conf_dict['learning_rate_initial']
    optimizer_type = conf_dict['optimizer_type']
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

train_loader, test_loader = load_dataset(input_dataset, img_size, batch_size, percentage)

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
for _, (x, y) in enumerate(test_loader):
    X_test, Y_test = x, y
Y_test = one_hot[Y_test]
del test_loader


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
        if pic_dim == 1:
            ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
        else:
            ax[i, j].imshow((test_images[k, :].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

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


def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc


start_epoch = 0
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_times'] = []
generator = Generator(z_dim, cond_num).to(device)
discriminator = Discriminator(pic_dim, cond_num).to(device)
loss_function = nn.BCELoss()
G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate_initial, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate_initial, betas=(0.5, 0.999))
if is_load and os.path.exists(result_dir + checkpoint_dir + '/checkpoint.pth.tar'):
    print('Loading existing models....')
    checkpoint = torch.load(result_dir + checkpoint_dir + '/checkpoint.pth.tar', map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    start_epoch = checkpoint['epoch']
    train_hist = checkpoint['train_hist']
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
else:
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

for epoch in range(start_epoch, train_epoch):
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()

    for idx, (feature, label) in enumerate(train_loader):
        batch_size = feature.size()[0]
        '''训练判别器'''
        # 评价所用标签
        y_real = torch.ones(batch_size)
        y_fake = torch.zeros(batch_size)
        y_real, y_fake = Variable(y_real.to(device)), Variable(y_fake.to(device))
        gen_fill = fill[label]
        feature, gen_fill = Variable(feature.to(device)), Variable(gen_fill.to(device))
        # 用真实图像训练
        # print('DIS_REAL ',idx)
        D_result = discriminator(feature, gen_fill).squeeze()
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
        D_train_loss = D_real_loss + D_fake_loss
        discriminator.zero_grad()
        D_fake_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.data.item())

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
        discriminator_copy = copy.deepcopy(discriminator)
        D_result = discriminator_copy(G_result, gen_fill).squeeze()
        # 计算损失
        G_train_loss = loss_function(D_result, y_real)
        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.item())
        del discriminator_copy

    pic_path = result_dir + images_dir + '/' + str(epoch + 1) + '.png'
    save_result((epoch + 1), path=pic_path)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    # 测试生成的图片训练效果
    val_z = torch.randn((generate_num, z_dim)).view(-1, z_dim, 1, 1)
    label = (torch.rand(generate_num, 1) * 10).type(torch.LongTensor).squeeze()
    val_label = one_hot[label]
    val_z, val_label = Variable(val_z.to(device)), Variable(val_label.to(device))
    with torch.no_grad():
        val_image = generator(val_z, val_label).view(generate_num, -1)
        classifier_NN = OneVsRestClassifier(MLPClassifier(random_state=2, alpha=1))
        NN_model2 = classifier_NN.fit(val_image.cpu(), val_label.cpu().view(generate_num, -1))
        predict_label = NN_model2.predict_proba(X_test.view(len(X_test), -1))
        false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(np.array(Y_test.view(len(Y_test), -1)),
                                                                               predict_label)
        res_MLP = [str(au) + " = " + str(roc_auc[au]) for au in roc_auc]
        fp = open(result_dir + checkpoint_dir + "/test_result.txt", 'a+')
        fp.write("Epoch {} ROC:".format(epoch + 1) + str(roc_auc["micro"]) + '\t')
        fp.write("MLP Result: " + str(res_MLP) + '\n')
        f.close()
        del classifier_NN, NN_model2, predict_label
    epoch_end_time = time.time()
    per_epoch_time = epoch_end_time - epoch_start_time
    print('Epoch %d - time: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), per_epoch_time, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
    print('ROC: ', str(roc_auc["micro"]))
    train_hist['per_epoch_times'].append(per_epoch_time)

print("Saving Models")
torch.save({
            'epoch': train_epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'D_optimizer': D_optimizer.state_dict(),
            'G_optimizer': G_optimizer.state_dict(),
            'train_hist': train_hist
            }, result_dir + checkpoint_dir + '/checkpoint.pth.tar')
save_train_hist(train_hist, path=result_dir + checkpoint_dir + '/train_hist.png')
