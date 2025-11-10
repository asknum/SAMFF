# SAMFF
环境配置
Step 1
conda create -n SAMFF python=3.8
conda activate SAMFF
Step 2
pip install -r requirements.txt

训练及测试命令
以GZ-CD为例
python train.py --config/gzcd.json 
python test.py --config/gzcd_test.json
测试权重及日志
通过网盘分享的文件：checkpoint.zip 链接: https://pan.baidu.com/s/19GdBflcV9WfxYMR_eTsRXQ?pwd=ysiy 提取码: ysiy
  
数据集格式：
GZ-CD
├── A
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── list
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
