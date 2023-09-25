## pre_data 文件夹
h52nii.ipynb 从h5，nii，pred这三个文件生成每个case单独存放的文件夹
show_data.ipynb 展示图片的工具集，包括直接plot和起来导出nii

## 训练环节
训练文件包含三个
#### train.py:
训练运行的文件，参数在最下方的dict中修改，不支持argparse，tranform也在这个文件中前部部分，优化器和lr_scheduler也在此定义
不支持直接python调用，如果单显卡也得用CUDA_VISIBLE_DEVICES 和 torchrun
#### model文件夹中的 medsam_trainer.py 文件：
主要的trainer文件，输入图片的大小，metric和loss在此定义。
写的时候略微仿照了pytorch lighting的思路，主要包含train_epoch和val_epoch函数
如果resume为None，会自动加载别人的checkpoint文件，并不是完全从零开始训练。
torch.compile是pytorch2的新功能，加速效果没有测试过，可以注释掉
#### base中的base_trainer文件：
logger,writer，save，load之类的功能，如果resume是None，则按日期创建文件夹，save的信息还有config和做的一些transform及随机增强的概率，因为我采用的是CyclicLR，所以学习率更新放在batch step里，如果其他scheduler得手动改
#### 另：train_decoder.py
train_decoder是基于编码后的embadding进行训练，无法做在线增强，弃用


## infer环节
基本所有推理文件主要由以下部分组成：
路径定义和模型加载
编码并保存
解码并保存
多框推理结果合成 或 多层合成

由于一层图片框数未知，没有强制对齐，所以编码batch size只能为1，可以用split参数把数据集拆分成多份，运行多次

### 2D infer
预先用utils.utils的generate_json生成json文件
test_decoder.py 基于h5文件中的embadding直接推理，如果有label还能算个label，很快，搭配pre_ssc.py使用
test.py 从图片开始推理，支持多框
inferance_no_npz_tutorial.ipynb 裁剪到中心再移回去的方法，已弃用
pre_data中的 infer_single.ipynb 单个nii文件的快速测试，可自定义框

### 3D infer
test_3d_no_singlkebox.py 一层图像对于一个bbox（中心的或者一整个），batch size可以任意
infer_3d_multibox.py 多个框推理，会保存embadding，根据split参数可以拆分数据集多次运行
iter_infer.py 基于infer_3d_multibox.py的结果从axis=1--2--0迭代推理


## utils文件夹：
dali_loader.py: 想加快cpu传gpu速度，没时间搞gpu的transform加速，所以transform还是在cpu做，后来发现用了更慢了，弃用
dataset.py：功能全用Transform搞定了，弃用
ddp_utils.py: DDP支持文件，加载DDP还有重写了下print和log info使其只在dist==0时输出
generate_aug.py 对test数据做shift用来做测试
mask_3d_process.ipynb: 开运算那节，推理后可以进行开运算处理
Transform.py： 极为重要，加载h5，z轴spacing统一，不同方式生成bbox等都在此
utils.py: 切片，timer装饰器等等小工具

### segment_anything文件夹： 模型存放地，不用动

### saved
模型参数和log存放地，默认保存近5个模型和最佳模型，最后一版是0818_98

### previous文件夹：基于npz格式的训练推理，已弃用



