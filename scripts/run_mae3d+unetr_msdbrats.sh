CUDA_VISIBLE_DEVICES=5 python main.py \
        configs/mae3d_msdbrats_1gpu.yaml \
	--cache_rate=1. \
	--epochs=500 \
	--mask_ratio=0.125 \
	--run_name='mae3d_vit_base_msd_brats_mr7o8_500e'
CUDA_VISIBLE_DEVICES=5 python main.py \
        configs/unetr_msdbrats_1gpu.yaml \
        --batch_size=6 \
	--epochs=300 \
        --lr=1.72e-2 \
        --cache_rate=0.5 \
        --run_name=unetr3d_vit_base_msd_brats_mr7o8_pre499_lrx0.5 \
        --pretrain=/nvme/zhoulei/ssl-framework/mae3d_vit_base_msd_brats_mr7o8_500e/ckpts/checkpoint_0499.pth.tar
