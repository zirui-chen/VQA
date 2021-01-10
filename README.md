# VQA

### 环境

- Ubuntu 任意版本操作系统

    > 数据下载需要wget命令，若无该条件，可自行去 https://visualqa.org/vqa_v1_download.html 下载对应数据集

- Python 3
- tensorflow 1.14.0



### 数据预处理

1. ##### 语言数据

    1. ##### 数据下载

        命令行cd进入本目录下的 `data/`，运行

        ```
        $ python vqa_preprocessing.py
        ```

        - 程序将从 https://visualqa.org/ 下载COCO 训练集+测试集数据，并在将文件解压在 `data/annotations` 路径下，得到5个文件 `mscoco_train2014_annotations.json`、`mscoco_val2014_annotations`、`MultipleChoice_mscoco_train2014_questions`、`MultipleChoice_mscoco_val2014_questions`、`MultipleChoice_mscoco_test2015_questions`
        - 合并5文件的训练数据为2个新训练集、测试集集合文件，并生成在路径下 `data/`，得到2个文件`vqa_raw_train.json`、`vqa_raw_test.json`

    2. ##### 获得问题数据特征

        返回项目根目录，运行

        ```
        $ python prepro.py
        ```

        在 `data/` 得到问题数据两种格式的特征文件 `data_prepro.h5` and `data_prepro.json`

2. ##### 图像数据

    项目根目录，运行

    ```
    $ th prepro_img.lua -input_json data_prepro.json -image_root path_to_image_root -cnn_proto path_to_cnn_prototxt -cnn_model path to cnn_model
    ```

    在根目录得到图像特征文件 `data_img.h5`



### 训练与预测

完成上述操作后，所有准备工作已就绪，现在已可进行训练及预测环节，仅需在根目录运行

```
$ python model_VQA.py
```

程序将依次执行 `train()` 训练模型并执行 `test()` 进行答案预测，程序将通过模型参数花费数小时进行训练，产生的模型保存在 `model_save/` 下，预测的答案结果保存在 `data.json` 中



### ← 运行截图具体见课程报告