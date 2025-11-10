<div align="center">

#  SAMFF

**Spatial Attention Multi-scale Feature Fusion for Change Detection**

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

##  目录

- [环境配置](#-环境配置)
- [训练及测试](#-训练及测试)
- [数据集格式](#-数据集格式)

---

##  环境配置

### Step 1: 创建环境

```bash
conda create -n SAMFF python=3.8
conda activate SAMFF
```

### Step 2: 安装依赖

```bash
pip install -r requirements.txt
```

---

##  训练及测试

### 训练命令

以 GZ-CD 数据集为例：

```bash
python train.py --config/gzcd.json
```

### 测试命令

```bash
python test.py --config/gzcd_test.json
```
### 测试权重及日志

```bash
通过网盘分享的文件：checkpoint.zip
链接: https://pan.baidu.com/s/1UL4lPr8oFomd25SX2yAlPA?pwd=tf5m 提取码: tf5m
```
### 参数说明

| 参数 | 描述 |
|------|------|
| `--config` | 配置文件路径 |

---

##  数据集格式

### 目录结构

数据集应按以下结构组织：

```
GZ-CD/
├── A/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

<div align="center">

** Remote Sensing Change Detection**

</div>
