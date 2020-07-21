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

##use gpu or not
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##set
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
pt = [640, 50, 640, 50]
labelmap = { 0: 'call', 1: 'fenxin', 2: 'normal', 3: 'smoke', 4: 'tired' }


##test_transform
transform_test = transforms.Compose([
     transforms.Resize([224, 384]), #注意testset不要randomresize
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
     ])

##rootdir and weights
rootdir = "/home/zigangzhao/DMS/dms0715/dms-5/video/convert/"
output_file = "/home/zigangzhao/DMS/dms0715/dms-5/video/output/"
weights = "/home/zigangzhao/DMS/dms0715/dms-5/checkpoint/resnet50/train/resnet50-42-best-0.9927184581756592.pth"

##network
print("---------start testing----------------")
resnet18 =  models.resnet18()
resnet18.fc = torch.nn.Linear(512, 5)
net = resnet18
net.load_state_dict(torch.load(weights))
net.eval()


##read video
for parent, dirnames, filenames in os.walk(rootdir):
	print(dirnames)
	print(filenames)
	for filename in filenames:
		# count += 1
		filename1 = os.path.splitext(filename)[0]
		os.mkdir(os.path.join(output_file,filename1))
		os.mkdir(os.path.join(output_file,filename1,'1'))
		os.mkdir(os.path.join(output_file,filename1,'2'))
		s1 = os.path.join(output_file,filename1,'1')
		print(s1)
		s2 = os.path.join(output_file,filename1,'2')
		print(s2)
		cap = cv2.VideoCapture(os.path.join(parent,filename))
		i = 0
		while(cap.isOpened()):
			ret, image_np = cap.read()
			if ret == False:
				break
			i += 1
            # print(i)

			# s=s1+"/"+filename1+'-'+str(i)+'.jpg'
			# s = s1+str(i)+'.jpg'
			# cv2.imwrite(s,frame)
			src_image_np = image_np.copy()
			img = Image.fromarray(image_np.astype('uint8')).convert('RGB') 
    
			image_np_expanded = transform_test(img).unsqueeze(0)
			# print(image_np_expanded.shape)
			image_np_expanded = image_np_expanded.to(device)

			# 执行侦测任务 
			net.eval()
			net = net.to(device)
			output = net(image_np_expanded)
			#print(output.shape)
			output = F.softmax(output, dim=1)
			#print(output)
			# 检测结果的可视化  
			frame, preds = torch.max(output, 1)
			# print(preds)

			cv2.rectangle(image_np, (int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])), 
									COLORS[1], 2)
			cv2.putText(image_np, labelmap[int(preds)], (int(pt[0]), int(pt[1])),
									FONT, 1, (255, 0, 255), 2, cv2.LINE_AA)  
		    
			s = s1 + "/" + filename1 + '-' + str(i) + '.jpg' 
			cv2.imwrite(s, src_image_np)	

			s = s2 + "/" + filename1 + '-' + str(i) + '.jpg'			    
			cv2.imwrite(s, image_np)

			# cv2.imshow('frame',frame)
			# s=s2+"/"+filename1+'-'+str(i)+'.jpg'
			# cv2.imwrite(s,frame)
			print(i)
		cap.release()
print("finished!")

