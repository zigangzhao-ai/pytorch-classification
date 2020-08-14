import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Config:

    log = "./log"  # Path to save log
    checkpoint_path = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"  # 从断点出重新加载模型，resume为模型地址
    evaluate = None  # 测试模型，evaluate为模型地址

    data_root = "/home/zigangzhao/DMS/dms0715/dms-5/data/0731/"
    train_dataset_path = os.path.join(data_root, 'train')
    val_dataset_path = os.path.join(data_root, 'test')

    seed = 0
    num_classes = 5
    input_image_size = 224
    # scale = 256 / 224
    EPOCH = 130
    # MILESTONES = [60, 120, 160]
    MILESTONES = [30, 60, 90, 110]
    # milestones = [30, 60, 90]
    # epochs = 150
    batch_size = 32
    accumulation_steps = 1
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 6
    print_interval = 1
    apex = False

    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomResizedCrop(input_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            #transforms.Resize(int(input_image_size * scale)),
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    loss_list = [
        {
            "loss_name": "Mixup_CELoss",
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "ce_family",
            "loss_rate_decay": "lrdv2"
        },
        {
            "loss_name": "KDLoss",
            "T": 1,
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "kd_family",
            "loss_rate_decay": "lrdv2"
        },

        {
            "loss_name": "GKDLoss",
            "T": 1,
            "loss_rate": 1,
            "factor": 1,
            "loss_type": "gkd_family",
            "loss_rate_decay": "lrdv2"
        },
        {
            "loss_name": "CDLoss",
            "loss_rate": 6,
            "factor": 0.9,
            "loss_type": "cd_family",
            "loss_rate_decay": "lrdv2"
        },
    ]