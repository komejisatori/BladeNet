## 针对局部运动模糊的去模糊网络BladeNet使用文档

### 1. 项目结构
```shell script
|-LocalBlurNet         #项目根目录
  |-config             #配置文件路径
  |-data               #dataloader部分
    |-dataloader.py    
  |-model              #训练模型保存目录
  |-network            #模型结构目录
    |-DeblurNet.py     #最终模型定义
    |-layers.py        #部分算子定义
    |-MaskNet.py       #模糊预测子网络定义
  |-result             #可视化结果默认保存目录
  |-criterion.py       #测试指标实现
  |-eval.py            #测试用脚本
  |-train.py           #训练用脚本
```

### 2. 环境配置
- Python=3.6
- pytorch=1.0.1
- visdom=0.1.8.9
- opencv-contrib-python=4.4.0.44  
- opencv-python=4.1.2.30

### 3. 数据准备
- 去模糊数据集
    
    将模糊图片和清晰图片放置在两个目录下，对应图片名称一致即可
    
- 合成数据

    合成数据和普通数据处理方法相同，如有模糊区域标注，也单独放在一个新目录下，并保持名字对应

### 2. 训练测试方法
- 构建配置文件
    
    配置文件默认在config路径下，每次运行后会自动备份当前的config文件。一份样例`config.yaml`文件如下：
    ```yaml
    date: 1217               #只作为实验备注，可任意修改，不影响功能
    gpu_available: '1,2'     #选用的GPU序号（训练支持多卡，测试单卡）
    gpu_num: 2               #需要和上方GPU数量一致
    onlyTrainMask: false     #是否只训练模糊区域预测子网络
    useMask: false           #是否使用模糊区域预测子网络
    fixMask: false           #是否固定模糊区域预测子网络
    usingSA: true            #是否使用空间注意力模块
    usingMaskLoss: false     #是否监督模糊区域预测子网络
    usingSALoss: true        #是否监督空间注意力模块
    finetune: false          #弃用
    local: false             #弃用
    other: ''                #只作为实验备注，可任意修改，不影响功能
    
    dataset: real            #只作为实验备注，可任意修改，不影响功能

    train_sharp: /path       #训练清晰数据的路径
    train_blur: /path        #训练模糊数据的路径
    
    train_sharp2: None       #带模糊区域标注的训练清晰数据的路径，不需要时写作None
    train_blur2: None        #带模糊区域标注的训练模糊数据的路径，不需要时写作None
    train_mask: None         #带模糊区域标注的训练模糊数据的路径，不需要时写作None
  
    test_sharp: /path        #测试用清晰数据的路径
    test_blur: /path         #测试用模糊数据的路径

    resize_size: 0           #弃用
    crop_size: 192           #训练数据裁剪的大小（长宽相同）
    model_dir: ./model       #保存模型的位置
    result_dir: ./result     #保存可视化结果的路径
    
    batchsize: 56            #数据batchsize

    save_epoch: 1            #保存模型频率
    lr: 0.001                #初始学习率
    step: [150, 200, 300]    #学习率衰减时间（gamma=0.5）
    
    mask_pretrained_model: None #单独加载去模糊部分，训练全部网络时使用，None则不加载
    sanet_pretrianed_model: None #单独加载模糊区域预测子网络，训练全部网络时使用，None则不加载
    
    pretrained_model: ./path #加载预训练模型的位置，None则不加载

    best_psnr: 0             #默认只会保存比这个值更高的模型
    startEpoch: 0            #开始时的epoch轮数，finetune时可以修改
    ```
  
 - 配置文件说明
 
    配置文件中3-6行的配置选项和实际模型对应关系如下
    
    |  配置项         | UNet  |SA-Unet  |Mask-Unet  |BladeNet  |MaskNet  |
    |  ----          | ----  |  ----   |----       |----      |----     |
    | onlyTrainMask  | ×     |  ×      |×          |×         |√        |
    | useMask        | ×     | ×       |  √        |√         |√        |
    | fixMask        | ×     |   ×     |×          |×         |×        |
    | usingSA        | ×     |    √    |×          |√         |×        |
    | usingMaskLoss  | ×     | √       |√          |√         |×        |
    | usingSALoss    | ×     | √       |×          |√         |×        |
    
    其中SA-代表空间注意力模块，Mask-代表模糊区域预测子网络，BladeNet代表整个网络，MaskNet代表只训练模糊区域子网络
    
 - 训练测试命令
 
    训练：`python train.py --c [PATH TO YAML FILE]`
    
    测试：`python eval.py --c [PATH TO YAML FILE]`