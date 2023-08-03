import numpy as np
import onnxruntime as ort
import torch
import cv2
def preict_one_img(img_path):
    img = cv2.imread(img_path) #读取图片
    img = cv2.resize(img, (64, 64))#调整图片尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 把图片BGR变成RGB
 
    img = np.transpose(img,(2,0,1))#调整维度将HWC - CHW
    img = np.expand_dims(img, 0) #添加一个维度 就是batch维度
    img = img.astype(np.float32)#格式转成float32
    img /= 255
   #调用onnxruntime run函数进行模型推理
    outputs = ort_session.run(
        None,
        {"input": img},
    )
    #outputs的输出类型为list类型，所以要先将list转换成numpy再转换成torch
    outputs1 = torch.from_numpy(np.array(outputs))
    #通过softmax进行最后分数的计算
    outputs_softmax = torch.softmax(outputs1[0], dim=1).numpy()[:, 0].tolist()
    print(outputs1)
   
if __name__ == '__main__':
    #cpu or gpu
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #onnx路径
    model_path = "weights/luosi.onnx"
    #加载onnx模型
    ort_session = ort.InferenceSession(model_path)
    # print(ort_session.get_inputs()[0].name)
    #图片路径
    i='data/classify/data_patch/val/pingtou/zhengchang_4_86.bmp'
    preict_one_img(i)