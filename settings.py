'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="DDP")

    parser.add_argument(
        '--hdf5_path',
        default="/data/sd0809/TianTanData/data_align_3mod.hdf5",
        type=str,
        help='Root directory path of data')
    
    parser.add_argument(
        '--json',
        default="./brats20.json",
        type=str,
        help='Root directory path of data')
    
    parser.add_argument(
        '--data_path',
        default="/data/sd0809/BraTS2020",
        type=str,
        help='Root directory path of data')
    
    # parser.add_argument(
    #     '--weight_path',
    #     default="/data/sd0809/Pretrain_Weight/brats/unet.pth",
    #     type=str)
    
    parser.add_argument(
        '--dice_model_path',
        default=None,
        type=str)
    
    parser.add_argument(
        '--hausdorff_model_path',
        default=None,
        type=str)

    parser.add_argument(
        '--dataset',
        default="tiantan",
        type=str,
        help='( tiantan | brats1617 | ...) '
    )

    parser.add_argument(
        '--pretrain_path',
        type=str,
        help=
        'Path for pretrained model.'
    )
    
    parser.add_argument(
        '--modality',
        nargs='+',
        type=str,
        help='modality needed')
    
    parser.add_argument(
        '--drop_path_rate',  # set to 0.001 when finetune
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续进")

    parser.add_argument(
        '--start_epoch',
        default=1,
        type=int,
        help='start epoch')
    
    parser.add_argument(
        '--optim',  # 分割任务的输出类别 WT, TC, ET
        default= 'adam',
        type=str,
        help='( sgd | adam | ...) '
    )

    parser.add_argument(
        '--lr_scheduler',  # 分割任务的输出类别 WT, TC, ET
        default= 'LambdaLR',
        type=str,
        help='( LambdaLR | StepLR | ExponentialLR | ReduceLROnPlateau ) '
    )

    parser.add_argument(
        '--loss_function',  # 分割任务的输出类别 WT, TC, ET
        default= 'Dice',
        type=str,
        help='( CE | Dice | DiceCE | ...) '
    )

    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=1e-3,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument(
        '--learning_rate_fate', 
        type=float,
        default=1e-2)
    
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-3)
    
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of jobs')
    
    parser.add_argument(
        '--batch_size', 
        default=1,
        type=int, 
        help='Batch Size')

    parser.add_argument(
        '--n_epochs',
        default=300,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--warmup_epochs',
        default=50,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--val_every',
        default=5,
        type=int,
        help='Number of total epochs to run')
    
    # model para
    parser.add_argument(
        '--model',
        default='unet',
        type=str,
        help='(resnet | unet | swin_unter')

    parser.add_argument(
        '--n_seg_classes',
        default=1,
        type=int,
        help='foreground and background')

    parser.add_argument(
        '--crop_H',
        default=256,
        type=int,
        help='Input size of depth')

    parser.add_argument(
        '--crop_W',
        default=256,
        type=int,
        help='Input size of height')

    parser.add_argument(
        '--crop_D',
        default=24,
        type=int,
        help='Input size of width')

    parser.add_argument(
        '--sw_batch_size',
        default=4,
        type=int,
        help='Input size of width')

    parser.add_argument(
        '--inf_overlap',
        default=0.5,
        type=float,
        help='Input size of width')

    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')

    parser.add_argument(
        '--save_folder', default="/data1/sd0809/output/", type=str, help='path to save model')
    
    parser.add_argument(
        '--test_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument(
        '--manual_seed', default=4294967295, type=int, help='Manually set random seed')

    args = parser.parse_args()
    
    return args