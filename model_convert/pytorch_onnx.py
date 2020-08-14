'''
code by zzg 2020-07-14
'''
##convert .pth model to onnx model

import torch
import torchvision.models as models
#
import os
import sys
# sys.path.append(os.getcwd())
sys.path.append("/home/zigangzhao/DMS/dms0715/dms-5/")
from models.resnet50 import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = torch.load("./resnet50-42-best-0.9927184581756592.pth") # pytorch模型加载

model = resnet50()
model.fc = torch.nn.Linear(2048, 5)
model.load_state_dict(torch_model) 

batch_size = 1  #批处理大小
input_shape = (3, 224, 384)   #输入数据

# # set the model to inference mode
#torch_model.eval()
model.eval()

x = torch.randn(batch_size, *input_shape)		# 生成张量
# x = x.to(device)
export_onnx_file = "0727.onnx"					# 目的ONNX文件名
torch.onnx.export( model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})
# x = torch.randn(batch_size,*input_shape)
# torch_out = torch.onnx._export(torch_model, x, "test.onnx", export_params=True)
print("finished!")
