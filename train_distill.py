# train.py
#!/usr/bin/env	python3

""" 
coede by zzg 2020-06-11
training by resnet50

"""

import os
import sys
import argparse
from datetime import datetime
from torch.backends import cudnn

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from conf import settings
from distillation.resnet_distillation import ChannelDistillResNet1834, ChannelDistillResNet1850

from torch.hub import load_state_dict_from_url
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from data_augmentation import mixup_data, mixup_criterion, LabelSmoothCEloss, cutmix
from distillation.conf.config import Config
from distillation import losses
from distillation.utils import adjust_loss_alpha


##set random seed
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def train(epoch):
    
    train_loss = 0.0 # cost function error
    loss = 0
    correct = 0.0
    ##use mixup
    ismixup = True
    iscutmix = False
    r = np.random.rand(1)

    loss_alphas = []
    for loss_item in Config.loss_list:
        loss_rate = loss_item["loss_rate"]
        factor = loss_item["factor"]
        loss_type = loss_item["loss_type"]
        loss_rate_decay = loss_item["loss_rate_decay"]
        loss_alphas.append(
            adjust_loss_alpha(loss_rate, epoch, factor, loss_type,
                              loss_rate_decay))

    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        # print(images, labels)
        if ismixup:
            images = images.cuda()
            labels = labels.cuda()
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)

            optimizer.zero_grad()

            stu_outputs, tea_outputs = net(inputs)

            for i, loss_item in enumerate(Config.loss_list):
                loss_type = loss_item["loss_type"]

                if loss_type == "ce_family":
                    tmp_loss1 = loss_alphas[i] * criterion[i](loss_function, stu_outputs[-1], targets_a, targets_b, lam)

                elif loss_type == "kd_family":
                    tmp_loss2 = loss_alphas[i] * criterion[i](stu_outputs[-1],
                                                            tea_outputs[-1])
                elif loss_type == "gkd_family":
                    tmp_loss3 = loss_alphas[i] * criterion[i](
                        stu_outputs[-1], tea_outputs[-1], labels)

                elif loss_type == "cd_family":
                    tmp_loss4 = loss_alphas[i] * criterion[i](stu_outputs[:-1],
                                                            tea_outputs[:-1])
       
            loss =  tmp_loss1 + tmp_loss2 + tmp_loss4
                       
            _, preds = stu_outputs[-1].max(1)
            correct += lam * preds.eq(targets_a).sum() + (1-lam) * preds.eq(targets_b).sum()
            
            loss.backward()
            optimizer.step()

        else:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            stu_outputs, tea_outputs = net(images)

            for i, loss_item in enumerate(Config.loss_list):
                loss_type = loss_item["loss_type"]

                if loss_type == "ce_family":
                    tmp_loss1 = loss_alphas[i] * criterion[i](stu_outputs[-1], labels)

                elif loss_type == "kd_family":
                    tmp_loss2 = loss_alphas[i] * criterion[i](stu_outputs[-1],
                                                            tea_outputs[-1])
                elif loss_type == "gkd_family":
                    tmp_loss3 = loss_alphas[i] * criterion[i](
                        stu_outputs[-1], tea_outputs[-1], labels)

                elif loss_type == "cd_family":
                    tmp_loss4 = loss_alphas[i] * criterion[i](stu_outputs[:-1],
                                                            tea_outputs[:-1])
       
            loss =  tmp_loss1 + tmp_loss2 + tmp_loss4
         
            _, preds = stu_outputs[-1].max(1)
            correct += preds.eq(labels).sum()

            loss.backward()
            optimizer.step()

        print('Training Epoch: [ {epoch} || [{trained_samples}/{total_samples}]\t || Loss: {:0.4f}\t || LR: {:0.6f} ]'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(training_loader.dataset)
        ))

        train_loss += loss.item()
    
    training_loss = train_loss / len(training_loader)
    training_acc = float(correct.float() / len(training_loader.dataset))
    loss_train.append(training_loss)
    acc_train.append(training_acc)
    #print(loss_train,acc_train)
    print('[train set: Average loss: {:.4f} || Accuracy: {:.4f}]'.format(
        training_loss, training_acc))
  

def eval_training(epoch):
    
    net.eval()
    
    loss = 0
    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
    
        images = images.cuda()
        labels = labels.cuda()
        
        stu_outputs, _ = net(images)

        _, preds = stu_outputs[-1].max(1)
        correct += preds.eq(labels).sum()
   
    print('[Accuracy: {:.4f}]'.format(correct.float() / len(test_loader.dataset)))
  
    # loss_test.append(test_loss / len(test_loader))
    acc_test.append(float(correct.float() / len(test_loader.dataset)))
    #print(loss_test,acc_test)
    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--numworks', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
   

    ##net--backbone
    # net = ChannelDistillResNet1834()
    net = ChannelDistillResNet1850()
    net = net.cuda()

    # print("load success!!")
    #data preprocessing:
    training_loader = get_training_dataloader(
        num_workers=args.numworks,
        batch_size=args.batch_size,
        shuffle=args.shuffle
        )
    
    test_loader = get_test_dataloader(
        num_workers=args.numworks,
        batch_size=args.batch_size,
        shuffle=args.shuffle
        )
    

    loss_function = nn.CrossEntropyLoss()
    loss_function1 = LabelSmoothCEloss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #### warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm if epoch <= args.warm else 0.5 * ( math.cos((epoch - args.warm) /(settings.EPOCH - args.warm) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
   
    ###add for distillation
    criterion = []
    for loss_item in Config.loss_list:
        loss_name = loss_item["loss_name"]
        loss_type = loss_item["loss_type"]
        if "kd" in loss_type:
            criterion.append(losses.__dict__[loss_name](loss_item["T"]).cuda())
        else:
            criterion.append(losses.__dict__[loss_name]().cuda())

    # print(criterion)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    print("start training!!")

    loss_train = []
    acc_train = []
    loss_test = []
    acc_test = []
    

    for epoch in range(1, Config.EPOCH+1):
        if epoch > args.warm:
            scheduler.step(epoch)

        train(epoch)
        # eval_training(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            best_acc = acc
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best-{}'.format(best_acc)))
           # torch.save(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best-{}'.format(best_acc)))          
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular-{}'.format(acc)))
            #torch.save(net, checkpoint_path.format(net=args.net, epoch=epoch, type='regular-{}'.format(acc)))



####plot 
ax1 = plt.subplot()
ax2 = ax1.twinx() #shared x axis with each other
ax1.plot(np.arange(1, len(loss_train) + 1), loss_train, color = 'g', label = 'train loss', linestyle = '-', linewidth = 2)
# ax1.plot(np.arange(1, len(loss_test ) + 1), loss_test, color = 'b', label = 'test loss', linestyle = '-', linewidth = 2)
ax2.plot(np.arange(1, len(acc_train) + 1), acc_train, color = 'g', label = 'train acc', linestyle = '-', linewidth = 2)
ax2.plot(np.arange(1, len(acc_test) + 1), acc_test, color = 'b', label = 'test acc', linestyle = '-', linewidth = 2)

ax1.legend(loc=(0.7, 0.7))  #使用(0.7,0.7)定义标签位置
ax2.legend(loc=(0.7, 0.5))
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.savefig("output/a.png", dpi = 400)
plt.show()

