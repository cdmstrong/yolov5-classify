"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
import torch.nn.functional as F
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
import pandas as pd
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU, Sigmod
from utils.augmentations import classify_transforms
from models.yolo import *
from utils.general import set_logging, check_img_size
import onnx
import onnxruntime
import numpy as np
import cv2
def get_out():
    # 加载模型

    model = onnx.load('weights/truck-cls.onnx')

    # 模型推理
    ori_output = copy.deepcopy(model .graph.output)
    # 输出模型每层的输出
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    ort_outs = ort_session.run(None, ort_inputs)
    #获取所有节点输出
    outputs = [x.name for x in ort_session.get_outputs()]
    # 生成字典，便于查找层对应输出
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    print("Mul_106")
    print(ort_outs["Mul_106"])
    # 创建一个空列表用于存储特定层的输出
layer_outputs = []



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/luosi.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img_size', nargs='+', type=int, default=[64, 64], help='image size')  # height, width
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    parser.add_argument('--onnx2pb', action='store_true', default=False, help='export onnx to pb')
    parser.add_argument('--onnx_infer', action='store_true', default=False, help='onnx infer test')
    #=======================TensorRT=================================
    parser.add_argument('--onnx2trt', action='store_true', default=False, help='export onnx to tensorrt')
    parser.add_argument('--fp16_trt', action='store_true', default=False, help='fp16 infer')
    #================================================================
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()
    # get_out()
    # Load PyTorch model
    model = attempt_load(opt.weights, device=torch.device('cpu'), inplace=True, fuse=True)  # load FP32 model
    # delattr(model.model[-1], 'anchor_grid')
    # model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers
    # model.model[-1].export_cat = True
    model.eval()
    labels = model.names
    print(f"{model.names}")
    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples
    # 注册钩子
    
    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    img = cv2.imread("data/classify/data_patch/val/M3-C/113443640ayopaujjr.jpg")
    img = cv2.resize(img, (64, 64))
    img = classify_transforms(64)(img)
    # print(model.model)
    # print(model.model)
    img = img.float().unsqueeze(0)
    # print(img)
    # img = torch.from_numpy(img/255).float().unsqueeze(0).permute(0, 3, 1, 2)
    y = model(img)  # dry run
    print(f"output :{y}")
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
            # elif isinstance(m.relu, nn.Sigmod):
            #     m.act = Sigmod()
        # elif isinstance(m, models.yolo.Detect): 
        #     m.forward = m.forward_export  # assign forward (optional)
        if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
                # if isinstance(m.relu, nn.Sigmod):
                #     m.branch2[i] = Sigmod()
            for i in range(len(m.branch2)):
                if isinstance(m.branch2[i], nn.SiLU):
                    m.branch2[i] = SiLU()
                # if isinstance(m.relu, nn.Sigmod):
                #     m.branch2[i] = Sigmod()
        if isinstance(m, models.common.BlazeBlock):#shufflenet block nn.SiLU
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
            # if isinstance(m.relu, nn.Sigmod):
            #     m.relu = Sigmod()
        if isinstance(m, models.common.DoubleBlazeBlock):#shufflenet block nn.SiLU
            if isinstance(m.relu, nn.SiLU):
                m.relu = SiLU()
            # if isinstance(m.relu, nn.Sigmod):
            #     m.relu = Sigmod()
            for i in range(len(m.branch1)):
                if isinstance(m.branch1[i], nn.SiLU):
                    m.branch1[i] = SiLU()
                # if isinstance(m.relu, nn.Sigmod):
                #     m.branch1[i] = Sigmod()
            # for i in range(len(m.branch2)):
            #     if isinstance(m.branch2[i], nn.SiLU):
            #         m.branch2[i] = SiLU()
    y = model(img)  # dry run
    providers =  ['CPUExecutionProvider']

    # ONNX export
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    input_names=['input']
    # session = onnxruntime.InferenceSession(f, providers=providers)
    # output_names = [output.name for output in session.get_outputs()]
    output_names = ["Mul_106"]
    # for output in session.get_outputs():
    #     print(output)
    #tensorrt 7
    # grid = model.model[-1].anchor_grid
    # model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
    #tensorrt 7

    torch.onnx.export(model, img, f, verbose=False, opset_version=12, 
        input_names=input_names,
        output_names=output_names,
        dynamic_axes = {'input': {0: 'batch'},
                        'output': {0: 'batch'}
                        } if opt.dynamic else None)
    # torch.onnx.export(model, img, f,
    #                   verbose=False,
    #                   opset_version=12,
    #                   input_names=['input'],
    #                   output_names=["det_stride_8", "det_stride_16", "det_stride_32"],
                    #   )  # for ncnn
    # model.model[-1].anchor_grid = grid
    #获取所有节点输出

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s' % f)
    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # onnx infer
    if opt.onnx_infer:
        im = img.cpu().numpy()#.astype(np.float32) # torch to numpy
        # y_onnx = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: im})[0]
        y_onnx = session.run(output_names, {session.get_inputs()[0].name: im})[0]
        print("output_names")
        print(output_names)
        print("yyyyyyyyyy")
        print(y)
        print(F.softmax(y, dim=1))
        print("y_onnx\n")
        print(y_onnx)
        print("Mul_106")
        # print(ort_outs["Mul_106"])
        # 输出每个层的输出
        for layer_output in y_onnx:
            print(layer_output)
        print("max(|torch_pred - onnx_pred|） =",abs(y.cpu().numpy()-y_onnx).max())

    # TensorRT export
    if opt.onnx2trt:
        from torch2trt.trt_model import ONNX_to_TRT
        print('\nStarting TensorRT...')
        ONNX_to_TRT(onnx_model_path=f,trt_engine_path=f.replace('.onnx', '.trt'),fp16_mode=opt.fp16_trt)

    # PB export
    if opt.onnx2pb:
        print('download the newest onnx_tf by https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf')
        from onnx_tf.backend import prepare
        import tensorflow as tf

        outpb = f.replace('.onnx', '.pb')  # filename
        # strict=True maybe leads to KeyError: 'pyfunc_0', check: https://github.com/onnx/onnx-tensorflow/issues/167
        tf_rep = prepare(onnx_model, strict=False)  # prepare tf representation
        tf_rep.export_graph(outpb)  # export the model

        out_onnx = tf_rep.run(img) # onnx output

        # check pb
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(outpb, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                input_x = sess.graph.get_tensor_by_name(input_names[0]+':0')  # input
                outputs = []
                for i in output_names:
                    outputs.append(sess.graph.get_tensor_by_name(i+':0'))
                out_pb = sess.run(outputs, feed_dict={input_x: img})

        print(f'out_pytorch {y}')
        print(f'out_onnx {out_onnx}')
        print(f'out_pb {out_pb}')

def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])