# CS3602 自然语言处理 2022秋季学期 口语语义解析大作业

### 创建环境

```bash
conda create -n slu python=3.6
source activate slu
pip install torch==1.9.1
pip install jieba==0.42.1
pip install transformers==4.5.1
pip install overrides==3.1.0
pip install elmoformanylangs==0.0.4.post2

```

根据 [ELMo代码仓库](https://github.com/HIT-SCIR/ELMoForManyLangs)指引，下载[简体中文模型](http://39.96.43.154/zhs.model.tar.bz2)，（亦可以直接使用此链接）后续根据`utils/args.py`中 `elmo_model`将`zhs.model`文件夹地址作为参数传入使用。

您亦可尝试以下指令安装所需环境。

```bash
pip install -r requirements.txt
```

### 运行

在根目录下运行

```bash
python scripts/slu_main.py
```

### 运行 Baseline (BiLSTM) 相关代码：

#### Embedding Layer:

-   使用预训练模型 roberta:
      ```bash
      python scripts/slu_main.py  --use_bert --alpha_filter
      ```
-   使用ELMo:

    ```bash
    python scripts/slu_main.py  --use_elmo 
    ```

#### Output Layer:

-   使用CRF:
  ```bash
  python scripts/slu_main.py  --use_crf
  ```

-   使用BiLSTM-LSTM的encoder-decoder模型:

    ```bash
    python scripts/slu_main.py  --use_lstm_decoder
    ```

-   使用增加了Focus Mechanism的BiLSTM-LSTM模型:
  ```bash
  python scripts/slu_main.py  --use_lstm_decoder --use_focus
  ```

#### 注意
-   可以同时使用任意一种 Embedding Layer 和 Output Layer 。
-   可以通过将软件包中：`elmoformanylangs/elmo.py` 中第95、96行注释减少ELMo无意义的info输出

### 运行Dual BiLSTM 相关代码：

-   手动融合：

    ```bash
     python scripts/slu_main.py  --algo Dual --rate_head 0.8 --rate_mid 0.6 --use_dict 
    ```
-   自动融合 By (Word Adaper):
    ```bash
     python scripts/slu_main.py  --algo Dual --Merge_Method Adapter --use_dict
    ```

-   Embedding Layer 选择ELMo, 因为其提供了词向量的提取。
-   可以运行任意一种Output Layer。
    -   LSTM Decoder 仅继承字级别的BiLSTM状态

### 测试与预测

```bash
 python scripts/slu_main.py --testing --<arg> <value>
```

-   注意： 测试模型需要添加训练中对应的args！
-   预测结果将被存放于 `args.dataroot/prediction.json`中。

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
  
    ```bash
    python scripts/slu_main.py --<arg> <value>
    ```
    
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
    
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU

+ `utils/vocab.py`:构建编码输入输出的词表

+ `utils/word2vec.py`:读取词向量

+ `utils/example.py`:读取数据

+ `utils/batch.py`:将数据以批为单位转化为输入

+ `model/slu_baseline_tagging.py`:Baseline模型

+ `scripts/slu_main.py`:主程序脚本

+ `model/slu_dual_tagging.py`：Dual 模型

+ `model/embed.py`: Embedding Layer

+ `model/Decode.py`不同Output Layer对应的不同Decoder以及Decode过程中需要的函数。



### 使用的预训练语言模型

-   Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext

-   ELMo: [simplified-Chinese](http://39.96.43.154/zhs.model.tar.bz2)

