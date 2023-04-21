CUDA_VISIBLE_DEVICES=5 python main.py \
	configs/mae3d_btcv_1gpu.yaml \
	--mask_ratio=0.125 \
	--run_name='mae3d_sincos_vit_base_btcv_mr125'
CUDA_VISIBLE_DEVICES=5 python main.py \
        configs/unetr_btcv_1gpu.yaml \
        --lr=3.44e-2 \
        --batch_size=6 \
        --run_name=unetr3d_vit_base_btcv_lr3.44e-2_mr125_10ke_pretrain_5000e \
        --pretrain=/nvme/zhoulei/ssl-framework/mae3d_sincos_vit_base_btcv_mr125/ckpts/checkpoint_9999.pth.tar
