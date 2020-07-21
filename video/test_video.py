# coding:utf-8
import numpy as np
import torch
import cv2
import os
import sys
sys.path.append("/home/zigangzhao/DMS/dms0715/dms-5/")

from torchvision import  models, transforms
import os.path
import torch.nn.functional as F
from PIL import Image
from models.resnet50 import resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
pt = [640, 50, 640, 50]
labelmap = { 0: 'call', 1: 'fenxin', 2: 'normal', 3: 'smoke', 4: 'tired' }


transform_test = transforms.Compose([
     transforms.Resize([224, 384]), #注意testset不要randomresize
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

weights = "/home/zigangzhao/DMS/dms0715/dms-5/checkpoint/resnet50/train/resnet50-42-best-0.9927184581756592.pth"
cap = cv2.VideoCapture("/home/zigangzhao/DMS/dms0715/dms-5/video/convert/hiv00033_20200709094310.mp4")
image = "/home/zigangzhao/DMS/dms0715/dms-5/data/0715/train/smoke/smoke_9.jpeg"


print("---------start testing----------------")
resnet50 =  models.resnet50()
resnet50.fc = torch.nn.Linear(2048, 5)
net = resnet50

net.load_state_dict(torch.load(weights))
net.eval()
##read video

while True:    
    ret, image_np = cap.read()

    img = Image.fromarray(image_np.astype('uint8')).convert('RGB') 
    
    image_np_expanded = transform_test(img).unsqueeze(0)
    print(image_np_expanded.shape)
    image_np_expanded = image_np_expanded.to(device)
    
    # image_np = cv2.resize(image_np, (384, 224), interpolation=cv2.INTER_CUBIC)  #resize(img,(W, H)) , interpolation=cv2.INTER_CUBIC
    # # print(image_np.shape)

    # # 扩展维度，应为模型 [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0) ##[1,224,384,3] NCHW
    # # print(image_np_expanded.shape)

    # image_np_expanded = torch.from_numpy(image_np_expanded) 
    # # print(image_np_expanded)
    # image_np_expanded = image_np_expanded.type(torch.FloatTensor).permute(0, 3, 1, 2) #[1,224,384,3] NHWC
    # print(image_np_expanded.shape)
    # image_np_expanded = image_np_expanded.to(device)
    # print(image_np_expanded)

    # 执行侦测任务 
    net.eval()
    net = net.to(device)
    output = net(image_np_expanded)
    #print(output.shape)
    output = F.softmax(output, dim=1)
    #print(output)
    # 检测结果的可视化  
    frame, preds = torch.max(output, 1)
    print(preds)

    cv2.rectangle(image_np, (int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])), 
                            COLORS[1], 2)
    cv2.putText(image_np, labelmap[int(preds)], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 0, 255), 2, cv2.LINE_AA)  

    print(preds)

    cv2.imshow('object class', cv2.resize(image_np,(1280, 720)))
    if cv2.waitKey(25) & 0xFF ==ord('q'):
        cv2.destroyAllWindows()
        break

print("finished!")


 