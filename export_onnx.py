import torch

# 实例你的模型
net = MyModel()
print('load static dict')
# 加载参数！ （如果忘记的话，，，就重来呗）
net.load_state_dict(torch.load("net.pth", map_location=torch.device('cpu')))
net.eval().cuda()
print('to cuda')

input1 = torch.randn(1, 3, 112,112).cuda()
input_names = [ "input"]
output_names = [ "output" ]
   
torch.onnx.export(net, input1, "out.onnx", verbose=True, input_names=input_names, output_names=output_names)

# 如果有多个输入可以用元组的形式包起来  如：
input2 = torch.randn(1,3,112,112).cuda()
torch.onnx.export(net, (input1,input2), "out.onnx", verbose=True, input_names=input_names, output_names=output_names)
