## LODE: Deep Local Deblurring and A New Benchmark [arxiv](https://arxiv.org/abs/2109.09149)

### 1. Code structure
```shell script
|-LocalBlurNet         
  |-config             #config files
  |-data               #dataloader
    |-dataloader.py    
  |-model              
  |-network            #network structure
    |-DeblurNet.py     
    |-layers.py        
    |-MaskNet.py       
  |-result             
  |-criterion.py       
  |-eval.py            
  |-train.py           
```

### 2. requirements
- Python=3.6
- pytorch=1.0.1
- visdom=0.1.8.9
- opencv-contrib-python=4.4.0.44  
- opencv-python=4.1.2.30


### 3. training configuration

    customizing your configuration file as follows:
    ```yaml
    date: 1217               
    gpu_available: '1,2'     #GPU IDS
    gpu_num: 2               #number of GPU in-use
    onlyTrainMask: false     
    useMask: false           
    fixMask: false           
    usingSA: true            
    usingMaskLoss: false     
    usingSALoss: true        
    finetune: false          
    local: false             
    other: ''                
    
    dataset: real            

    train_sharp: /path       
    train_blur: /path        
    
    train_sharp2: None       
    train_blur2: None        
    train_mask: None         
  
    test_sharp: /path        
    test_blur: /path         

    resize_size: 0           
    crop_size: 192           
    model_dir: ./model       
    result_dir: ./result     
    
    batchsize: 56            

    save_epoch: 1            
    lr: 0.001                
    step: [150, 200, 300]    
    
    mask_pretrained_model: None 
    sanet_pretrianed_model: None 
    
    pretrained_model: ./path 

    best_psnr: 0             
    startEpoch: 0            
    ```
  
 - module selection
 
    the relationship between configuration file and networks in paper
    
    |  configurations         | UNet(BladeNet-)  |BSA only  |LBP only  |BladeNet  |
    |  ----          | ----  |  ----   |----       |----      |
    | onlyTrainMask  | ×     |  ×      |×          |×         |
    | useMask        | ×     | ×       |  √        |√         |
    | fixMask        | ×     |   ×     |×          |×         |
    | usingSA        | ×     |    √    |×          |√         |
    | usingMaskLoss  | ×     | √       |√          |√         |
    | usingSALoss    | ×     | √       |×          |√         |
    
    
 - experiment scripts
 
    for training：`python train.py --c [PATH TO YAML FILE]`
    
    for testing：`python eval.py --c [PATH TO YAML FILE]`
