'''训练'''：
  python classify/train.py --model weights/yolov5s-cls.pt --data data/classify/data_patch --epochs 100 --imgsz 64 --device 1

# 训练数据分成如下
  data_patch
    ---train
        --- class1
        --- class2
        --- class3
    ---val
        --- class1
        --- class2
        --- class3
# 网络结构
特征提取使用的是 v5的特征提取
最后加上 classify 自定义的 分类网络

# 使用配置文件yaml 生成网络
默认配置文件为
- yolov5s-cls.yaml

修改了原有的v5 结构，保留backbone部分， 在head 新增Classify 层

# 使用预训练模型生成分类模型
原理： 提取模型的backbone部分， 替换head部分为分类层
