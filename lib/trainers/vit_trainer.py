import os
import math

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import sys
sys.path.append('..')

import models
import networks
from utils import get_conf, SmoothedValue, concat_all_gather, compute_aucs, LayerDecayValueAssigner
from datasets import ImageListDataset
import data_preprocessing

import wandb

from .base_trainer import BaseTrainer

from timm.data.auto_augment import rand_augment_transform
from timm.data import Mixup
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy

from collections import defaultdict

# def bce_loss(pred, label):
#     loss = F.binary_cross_entropy_with_logits(pred, label, None,
#                                             pos_weight=None,
#                                             reduction='mean')
#     return loss

# class BCELoss(torch.nn.Module):
#     def forwrad(self, pred, label):
#         return F.binary_cross_entropy_with_logits(pred, label, None,
#                                             pos_weight=None,
#                                             reduction='mean')

class VitTrainer(BaseTrainer):
    r"""
    Vit Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            if args.dataset == 'im100':
                num_classes = 100
            elif args.dataset == 'im1k':
                num_classes = 1000
            elif args.dataset == 'cxr14':
                num_classes = 14
            else:
                raise ValueError(f"Unsupported dataset {args.dataset}")

            # setup mixup and loss functions
            if args.mixup > 0:
                self.mixup_fn = Mixup(
                            mixup_alpha=args.mixup, 
                            cutmix_alpha=args.cutmix, 
                            label_smoothing=args.label_smoothing, 
                            num_classes=num_classes)
                if args.dataset == 'cxr14':
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
                else:
                    self.loss_fn = SoftTargetCrossEntropy()
            else:
                self.mixup_fn = None
                if args.dataset == 'cxr14':
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
                else:
                    self.loss_fn = torch.nn.CrossEntropyLoss()

            self.model = getattr(models, self.model_name)(
                                                        num_classes=num_classes,
                                                        drop_path_rate=args.drop_path,
                                                        use_learnable_pos_emb=True)

            # load pretrained weights
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                for key in list(state_dict.keys()):
                    if key.startswith('encoder.'):
                        state_dict[key[len('encoder.'):]] = state_dict[key]
                        del state_dict[key]
                # self.model.load(state_dict)
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        # model = self.wrapped_model
        model = self.model

        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        # optim_params = self.group_params(model)
        optim_params = self.get_parameter_groups(get_layer_id=assigner.get_layer_id, get_layer_scale=assigner.get_scale)
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    def get_vit_scratch_augmentation(self):
        raise NotImplementedError("Vit scratch augmentation has not been implemented.")

    def get_vit_ft_augmentation(self):
        args = self.args
        if args.mean_std_type == 'IMN':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif args.mean_std_type == 'MED':
            mean = (0.5, 0.5, 0.5)
            std = (0.225, 0.225, 0.225)
        else:
            raise ValueError(f"Unsuported mean_std_type {args.mean_std_type}")

        self.mean, self.std = mean, std
        normalize = transforms.Normalize(mean=mean, std=std)

        aa_params = dict(
            translate_const=int(args.input_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        augmentation = [
                transforms.RandomResizedCrop(args.input_size, scale=(args.crop_min, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform(config_str=args.randaug, hparams=aa_params),
                transforms.ToTensor(),
                normalize
            ]
        return augmentation

    def get_vit_val_augmentation(self):
        args = self.args
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        augmentation = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        return augmentation

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args
            if 'ft' in args.proj_name:
                augmentation = self.get_vit_ft_augmentation()
            elif 'scratch' in args.proj_name:
                augmentation = self.get_vit_scratch_augmentation()
            else:
                raise NotImplementedError(f"augmentation required by project {args.proj_name} is not implemented.")

            if args.dataset == 'cxr14':
                multiclass = True
            else:
                multiclass = False

            train_dataset = ImageListDataset(
                data_root=args.data_path,
                listfile=args.tr_listfile,
                transform=transforms.Compose(augmentation),
                nolabel=False,
                multiclass=multiclass)

            # for evaluation
            val_augmentation = self.get_vit_val_augmentation()
            
            val_dataset = ImageListDataset(
                data_root=args.data_path,
                listfile=args.va_listfile,
                transform=transforms.Compose(val_augmentation),
                nolabel=False,
                multiclass=multiclass)

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None

            self.dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=(train_sampler is None),
                                                        num_workers=self.workers, 
                                                        pin_memory=True, 
                                                        sampler=train_sampler, 
                                                        drop_last=True)
            self.iters_per_epoch = len(self.dataloader)

            self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                        batch_size=int(1.5 * self.batch_size), 
                                                        shuffle=(val_sampler is None),
                                                        num_workers=self.workers, 
                                                        pin_memory=True, 
                                                        sampler=val_sampler,
                                                        drop_last=False)
            self.val_iters = len(self.val_dataloader)
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)

            # if epoch == args.start_epoch:
            #     print("=> First Evaluation")
            #     self.evaluate(epoch=epoch, niters=niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            # evaluate after each epoch training
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                self.evaluate(epoch=epoch, niters=niters)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()

        for i, (image, target) in enumerate(train_loader):
            # adjust learning at the beginning of each iteration
            # print("start adjusting learning rate")
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)
            # print("finish adjusting learning rate")

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if mixup_fn is not None:
                # print("start setting mixup")
                image, target = mixup_fn(image, target)
                # print("finish setting mixup")

            # compute output and loss
            with torch.cuda.amp.autocast(True):
                # print("start training for one batch")
                loss = self.train_class_batch(model, image, target, loss_fn)
                # print("finish training for one batch")

            # compute gradient and do SGD step
            # print("start computing gradient and updating parameters")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print("finish computing gradient and updating parameters")

            # torch.cuda.synchronize()

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      f"PeRate: {model.module.pe_rate:.05f} | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": last_layer_lr,
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
        return niters

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = model(samples)
        loss = criterion(outputs, target)
        return loss

    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        if args.eval_metric == 'acc':
            criterion = torch.nn.CrossEntropyLoss()
        elif args.eval_metric == 'auc':
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Only support acc and auc for now")
        meters = defaultdict(SmoothedValue)

        # switch to evaluation mode
        model.eval()

        if args.eval_metric == 'auc':
            pred_list = []
            targ_list = []

        for i, (image, target) in enumerate(val_loader):
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(image)
                loss = criterion(output, target)

            if args.eval_metric == 'acc':
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                batch_size = image.size(0)
                meters['loss'].update(value=loss.item(), n=batch_size)
                meters['acc1'].update(value=acc1.item(), n=batch_size)
                meters['acc5'].update(value=acc5.item(), n=batch_size)

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.val_iters} | "
                      f"Loss: {loss.item():.03f} | "
                      f"Acc1: {acc1.item():.03f} | "
                      f"Acc5: {acc5.item():.03f}")
            elif args.eval_metric == 'auc':
                batch_size = image.size(0)
                meters['loss'].update(value=loss.item(), n=batch_size)
                pred_list.append(concat_all_gather(output, args.distributed))
                targ_list.append(concat_all_gather(target, args.distributed))

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.val_iters} | "
                      f"Loss: {loss.item():.03f}")
            else:
                raise NotImplementedError("Only support Acc and AUC for now.")

                    # gather the stats from all processes
        if args.distributed:
            for k, v in meters.items():
                print(f'====> start synchronizing meter {k}...')
                v.synchronize_between_processes()
                print(f'====> finish synchronizing meter {k}...')

        # compute auc
        if args.eval_metric == 'auc':
            pred_array = torch.cat(pred_list, dim=0).data.cpu().numpy()
            targ_array = torch.cat(targ_list, dim=0).data.cpu().numpy()
            auc_list, mAUC = compute_aucs(pred_array, targ_array)

            print(f"==> Epoch {epoch:04d} test results: \n"
                  f"=> mAUC: {mAUC:.05f}")

            if args.rank == 0:
                wandb.log(
                    {
                     "Eval Loss": meters['loss'].global_avg,
                     "mAUC": mAUC
                    },
                    step=niters,
                )
        elif args.eval_metric == 'acc':
            print(f"==> Epoch {epoch:04d} test results: \n"
                  f"=>  Loss: {meters['loss'].global_avg:.05f} \n"
                  f"=> Acc@1: {meters['acc1'].global_avg:.05f} \n"
                  f"=> Acc@5: {meters['acc5'].global_avg:.05f} \n")

            if args.rank == 0:
                wandb.log(
                    {
                     "Eval Loss": meters['loss'].global_avg,
                     "Acc@1": meters['acc1'].global_avg,
                     "Acc@5": meters['acc5'].global_avg
                    },
                    step=niters,
                )

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler']) # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
    
    def adjust_posemb_rate(self, epoch, args):
        """Base schedule: Cosine Increase"""
        pe_rate = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        return pe_rate