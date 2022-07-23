import torch
from micronet_det import MicroNet
from torchsummary import summary
from collections import OrderedDict
import torchvision
a=torch.randn(4,3,224,224)
model_locate=r'C:\Users\Administrator\Desktop\micronet-main\backbone\micronet-m3.pth'
micro_model_param=torch.load(model_locate,map_location=torch.device('cuda'))
micro_model=MicroNet()
# ans=micro_model(a)
# print(ans.shape)

new_state_dict = OrderedDict()
for k, v in micro_model_param.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
micro_model.load_state_dict(new_state_dict,strict=True)
input_names = ['input']
output_names = ['output0','output1','output2','output3']
# print("summary",summary(b,(3,224,224)))

# torch.onnx.export(micro_model, a, './micronet_pretrain.onnx'
#                   , input_names=input_names
#                   , output_names=output_names
#                   , verbose=True
#                   ,opset_version=9
#                   ,do_constant_folding=True
#                   ,dynamic_axes={"input": {0: 'batch'}})

ans=micro_model(a)
for i,v in enumerate(ans):
  print('i=',i,'v=',v.shape)
