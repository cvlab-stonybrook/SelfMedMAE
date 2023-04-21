from re import L
import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d

def get_scratch_train_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1, neg=1, num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z), random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"])
                # transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_mae_pretrain_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1, neg=1, num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_val_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_vis_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                # transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_raw_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.AddChanneld(keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')
        # import pdb
        # pdb.set_trace()
        return transform(img)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    elif args.dataset == 'msd_brats':
        post_pred = transforms.Compose([transforms.EnsureType(), transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        post_label = transforms.Identity()
    return post_pred, post_label