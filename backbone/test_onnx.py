import onnxruntime as ort
import numpy as np
onnx_model_path = "./micronet_pretrain.onnx"  # onnx模型
ort_session = ort.InferenceSession(onnx_model_path)
# 输入层名字
onnx_input_name = ort_session.get_inputs()[0].name
# 输出层名字
onnx_outputs_names=[]
for x in ort_session.get_outputs():
  onnx_outputs_names.append(x.name)

print(onnx_outputs_names)
img=np.random.rand(1,3,224,224)
# 如果是RGB则
# img = img[np.newaxis, :, :, :]
input_blob = np.array(img, dtype=np.float32)
onnx_result = ort_session.run(onnx_outputs_names, input_feed={onnx_input_name: input_blob})
for k in onnx_result:
  print(np.shape(k))

# print(idx)  # 打印识别结果
# print(res[idx])  # 对应的概率
