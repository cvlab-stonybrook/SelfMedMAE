# SelfMedMAE: Self Pre-training with Masked Autoencoders for Medical Image Analysis

### Preparation

1. Install PyTorch, timm and [MONAI](https://monai.io/index.html).
3. Download the [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) and [MSD_BraTS](http://medicaldecathlon.com/) data.
4. Install Wandb for logging and visualizations.

### Stage 1: MAE Pre-Training
The run scripts are in directory scripts
```
python main.py \
        configs/mae3d_btcv_1gpu.yaml \
        --mask_ratio=0.125 \
        --run_name='mae3d_sincos_vit_base_btcv_mr125'
```
The default configurations are set in `configs/mae3d_btcv_1gpu.yaml`. You can overwrite the configurations by passing arguments with the corresponding key names through the command line, e.g., `mask_ratio`. We use Wandb to monitor the training process and visualize the masked reconstruction. During the training, the output including checkpoints and Wandb local files are all stored in the specified `output_dir` value in the configurations.
The core MAE codes locate in `lib/models/mae3d.py`.

### Stage 2: UNETR Fine-tuning
The run scripts is in directory scripts
```
python main.py \
        configs/unetr_btcv_1gpu.yaml \
        --lr=3.44e-2 \
        --batch_size=6 \
        --run_name=unetr3d_vit_base_btcv_lr3.44e-2_mr125_10ke_pretrain_5000e \
        --pretrain=$YOUR Pre-Trained MAE Checkpoint$
```
The core UNETR codes locate in `lib/models/unetr3d.py`.
