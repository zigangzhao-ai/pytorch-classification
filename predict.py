"""
code by zzg 2020-07-01
predict single image
"""
##test
###预测2类示例

try:
    import xml.etree.cElementTree as ET  
except ImportError:
    import xml.etree.ElementTree as ET

import torch
import os
import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from conf import settings
import torchvision.transforms as transforms
import torchvision.models as models

mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def predict(model):
    # 读入模型
    resnet18 = models.resnet50()
    resnet18.fc = torch.nn.Linear(2048,2)
    net = resnet18
    
      #load_checkpoint
    net.load_state_dict(torch.load(model))
  
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        net.cuda()
    pred_list, _id = [], []

    print("-----starting predicting!------------")

    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        img = Image.open(img_path).convert('RGB')
        path_xml = img_path.replace(".jpeg", ".xml")
        # print(type(img))
        tree = ET.ElementTree(file=path_xml)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        ObjBndBoxSet = []
        for Object in ObjectSet:
            # ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)#-1 
            y1 = int(BndBox.find('ymin').text)#-1
            x2 = int(BndBox.find('xmax').text)#-1
            y2 = int(BndBox.find('ymax').text)#-1
            BndBoxLoc = [x1,y1,x2,y2]
            # print(x1,y1,x2,y2)
            ObjBndBoxSet.append(BndBoxLoc) 
        x1,y1,x2,y2 = ObjBndBoxSet[0]

        img = np.array(img)
        img1 = img[y1:y2, x1:x2]
        img1 = Image.fromarray(img1.astype('uint8')).convert('RGB') 
        img1 = get_test_transform()(img1).unsqueeze(0)

        if torch.cuda.is_available():
            img1 = img1.cuda()
        with torch.no_grad():
            out = net(img1)
            print(out)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list



if __name__ == "__main__":

    trained_model = "/home/zigangzhao/DMS/dms-2/checkpoint/resnet50/2020-07-01T14:27:31.985135/resnet50-60-regular-1.0.pth"
    model_name = "resnet50"
    classname = { 0:'noraml', 1:'tired'}
    img_Lists = glob.glob(settings.TSET_IMAGE + '/*.jpeg')
    print(img_Lists)
    imgs = img_Lists


    _id, pred_list = predict(trained_model)
    print(classname)
    print(_id, pred_list)
    for i in range(len(pred_list)):
        print("{} --> {}".format(_id[i],classname[pred_list[i]]))

    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv(settings.BASE + '{}_submission.csv'
                      .format(model_name), index=False, header=False)
