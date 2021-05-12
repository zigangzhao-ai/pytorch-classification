#test.py
#!/usr/bin/env python3

"""
test neuron network performace
print top1 and top5 err on test dataset of a model

add precision and recall as evaluation index
code by zzg 2020-06-11
"""

import argparse
#from dataset import *
#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--num', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    resnet18 = models.resnet50()
    resnet18.fc = torch.nn.Linear(2048, 5)
    net = resnet18

    # net = get_network(args)
    test_loader = get_test_dataloader(
        num_workers=args.num,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()
    net = net.cuda()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    predlist1 = torch.zeros(0, dtype=torch.long, device='cpu')

    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    label_names = ["call","fenxin","normal","smoke","tired"]

    for n_iter, (image, label) in enumerate(test_loader):
        # print("iteration: {}\t total {} iterations".format(n_iter + 1, len(test_loader)))
        image = image.cuda()
        label0 = label.cuda()
        output = net(image)
        # print(output)
        # print(label0)
        _, pred = output.topk(5, 1, largest=True, sorted=True)
        # print(pred)

        label = label0.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1 
        correct_1 += correct[:, :1].sum()

        _, preds = torch.max(output, 1)
        # print(preds)
        predlist = torch.cat([predlist, preds.view(-1).cpu()])
        # predlist1 = torch.cat([predlist1, output.view(-1).cpu()])
        lbllist = torch.cat([lbllist, label0.view(-1).cpu()])

    report = classification_report(lbllist.numpy(), predlist.numpy(), target_names=label_names,
                                   digits=2)
  
    print("Top 1 err: {:.3}%".format((1 - correct_1 / len(test_loader.dataset))*100))
    print("Top 5 err: {:.3}%".format((1 - correct_5 / len(test_loader.dataset))*100))
    print(report)

    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
