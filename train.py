import os
import argparse
import time
import shutil
import csv
import yaml

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision.transforms import v2

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyOHEMLoss, BoundaryAwareLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Parse training options for Segmentation Experiments."""
    parser = argparse.ArgumentParser(description='Boundary-Aware Fast-SCNN on PyTorch')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    # Default to local empty folders
    parser.add_argument('--data_path', type=str, default='./datasets/dataset_demo', help='Path to dataset directory')
    
    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    # Model and dataset configurations
    parser.add_argument('--model', type=str, default=config.get('model', {}).get('name', 'fast_scnn'),
                        help='model name')
    parser.add_argument('--dataset', type=str, default=config.get('dataset', {}).get('name', 'citys'),
                        help='dataset name')
    parser.add_argument('--dataset_type', type=str, default=config.get('dataset', {}).get('dataset_type', 'custom_agricultural'),
                        help='dataset type (e.g. public_cityscapes, custom_agricultural)')
    parser.add_argument('--base-size', type=int, default=config.get('dataset', {}).get('base_size', 1024),
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=config.get('dataset', {}).get('crop_size', 1024),
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    
    # Training hyperparameters
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=config.get('training', {}).get('epochs', 300), metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=config.get('training', {}).get('batch_size', 8),
                        metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=config.get('training', {}).get('learning_rate', 0.01), metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.get('training', {}).get('weight_decay', 0.001),
                        metavar='M', help='weight decay')
    parser.add_argument('--optimizer', type=str, default=config.get('training', {}).get('optimizer', 'AdamW'),
                        choices=['AdamW', 'SGD'], help='optimizer type (AdamW or SGD)')
    
    # Loss settings
    loss_cfg = config.get('loss', {})
    parser.add_argument('--use_biou', type=bool, default=loss_cfg.get('use_biou', True),
                        help='use Boundary Intersection over Union (BIoU) loss')
    parser.add_argument('--dilation_ratio', type=float, default=loss_cfg.get('dilation_ratio', 0.02),
                        help='dilation ratio for BIoU boundary map')
    parser.add_argument('--max_biou_ratio', type=float, default=loss_cfg.get('max_biou_ratio', 0.5),
                        help='Maximum ratio of BIoU loss relative to total loss')

    # Checkpoint settings (default to local empty folder)
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--weight-path', default='./weights/weight_demo.pth',
                        help='Path to checkpoint model for evaluation')
    
    # Evaluation settings
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Data augmentation and transforms
        input_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        ])

        data_kwargs = {
            'root': args.data_path,
            'transform': input_transform, 
            'base_size': args.base_size, 
            'crop_size': args.crop_size,
            'dataset_type': args.dataset_type
        }
        
        # Initialize full dataset and split into train/val subsets
        self.full_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)

        self.dataset_size = len(self.full_dataset)
        indices = list(range(self.dataset_size))
        
        if self.dataset_size < 2:
            train_idx = indices
            val_idx = indices
        else:
            test_size = max(1, int(self.dataset_size * 0.2)) if self.dataset_size >= 5 else 1
            train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)

        train_dataset = Subset(self.full_dataset, train_idx)
        val_dataset = Subset(self.full_dataset, val_idx)

        train_dataset.num_class = self.full_dataset.num_class
        val_dataset.num_class = self.full_dataset.num_class

        # Create dataloaders
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

        # Create Fast-SCNN network
        # Since train.py uses --resume to load weights manually if needed, we don't pass weight_path here.
        self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(args.device)

        # Resume from checkpoint if specified
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or ext == '.pth', 'Only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # Setup Loss Criterion
        base_criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight, ignore_index=-1)
        if args.use_biou:
            self.criterion = BoundaryAwareLoss(base_loss_fn=base_criterion, dilation_ratio=args.dilation_ratio).to(args.device)
        else:
            self.criterion = base_criterion.to(args.device)

        # Setup Optimizer and Learning Rate Scheduler
        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args.lr,
                                             momentum=0.9,
                                             weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)

        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # Setup Evaluation Metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def train(self):
        """Training loop."""
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            evaluation_iter = [0,0,0,0,0,0] if not self.args.use_biou else [0,0,0,0,0,0,0]
            iter_num = 0

            for i, (images, targets) in enumerate(self.train_loader):
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # Forward pass
                outputs = self.model(images)

                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                self.metric.update(pred, targets.cpu().numpy())
                pixAcc, MIoU, IoU = self.metric.get()

                # Compute loss (with or without BIoU)
                if self.args.use_biou:
                    biou_ratio = min(self.args.max_biou_ratio, epoch/self.args.epochs * self.args.max_biou_ratio)
                    loss, mbiou = self.criterion(outputs, targets, biou_ratio=biou_ratio)
                else:
                    loss = self.criterion(outputs, targets)
                    mbiou = 0.0

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    if self.args.use_biou:
                        evaluation_iter = [x + y for x, y in zip(evaluation_iter, [pixAcc, IoU[0], IoU[1], MIoU, loss.item(), float(mbiou)])]
                        print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f || BIoU: %.4f' % (epoch, self.args.epochs, i + 1, len(self.train_loader), time.time() - start_time, cur_lr, loss.item(), mbiou))
                    else:
                        evaluation_iter = [x + y for x, y in zip(evaluation_iter, [pixAcc, IoU[0], IoU[1], MIoU, loss.item()])]
                        print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (epoch, self.args.epochs, i + 1, len(self.train_loader), time.time() - start_time, cur_lr, loss.item()))
                    iter_num+=1

            # Validation logic
            if self.args.no_val:
                save_checkpoint(self.model, self.args, is_best=False)
            else:
                mean_training_eval = [x / iter_num for x in evaluation_iter] if iter_num > 0 else evaluation_iter
                print('Train evaluation:')
                if self.args.use_biou:
                    print('Epoch: %2d || pixAcc %.4f || IoU(unlabel) %.4f || IoU(road) %.4f || MIoU %.4f || Loss %.4f || mBIoU %.4f\n' % (epoch, *mean_training_eval[:6]))
                else:
                    print('Epoch: %2d || pixAcc %.4f || IoU(unlabel) %.4f || IoU(road) %.4f || MIoU %.4f || Loss %.4f\n' % (epoch, *mean_training_eval[:5]))
                
                train_evaluate_value.append(([epoch] + mean_training_eval))
                self.validation(epoch)

            # Save training evaluation metrics to CSV
            with open(os.path.join(evaluate_path, 'train_evaluate.csv'), 'w', newline='') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL,delimiter=',')
                writer.writerows(train_evaluate_value)

            # Save validation evaluation metrics to CSV
            with open(os.path.join(evaluate_path, 'val_evaluate.csv'), 'w', newline='') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL,delimiter=',')
                writer.writerows(val_evaluate_value)

    def validation(self, epoch):
        """Validation loop."""
        print('Validating...')
        is_best = False
        self.metric.reset()
        self.model.eval()

        evaluation = [0,0,0,0,0,0] if not self.args.use_biou else [0,0,0,0,0,0,0]
        sample = 0

        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                image = image.to(self.args.device)
                target = target.to(self.args.device)

                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                self.metric.update(pred, target.cpu().numpy())
                pixAcc, MIoU, IoU = self.metric.get()
                
                if self.args.use_biou:
                    biou_ratio = min(self.args.max_biou_ratio, epoch/self.args.epochs * self.args.max_biou_ratio)
                    loss, mbiou = self.criterion(outputs, target, biou_ratio=biou_ratio)
                    evaluation = [x + y for x, y in zip(evaluation, [epoch, pixAcc, IoU[0], IoU[1], MIoU, loss.item(), float(mbiou)])]
                else:
                    loss = self.criterion(outputs, target)
                    evaluation = [x + y for x, y in zip(evaluation, [epoch, pixAcc, IoU[0], IoU[1], MIoU, loss.item()])]
                sample += 1
        
        # Save best model
        new_pred = (pixAcc + MIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            save_checkpoint(self.model, self.args, is_best=is_best)
            
        mean_evaluation = [x / sample for x in evaluation] if sample > 0 else evaluation
        
        if self.args.use_biou:
            print('Epoch: %2d || pixAcc %.4f || IoU(unlabel) %.4f || IoU(road) %.4f || MIoU %.4f || Loss %.4f || mBIoU %.4f\n' % tuple(mean_evaluation[:7]))
        else:
            print('Epoch: %2d || pixAcc %.4f || IoU(unlabel) %.4f || IoU(road) %.4f || MIoU %.4f || Loss %.4f\n' % tuple(mean_evaluation[:6]))
            
        val_evaluate_value.append(mean_evaluation)

def save_checkpoint(model, args, is_best=False):
    """Save model checkpoint."""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(save_path, best_filename)

if __name__ == '__main__':
    evaluate_path = os.path.join('.', 'evaluation', 'Evaluation data')
    if not os.path.isdir(evaluate_path):
        os.makedirs(evaluate_path)

    args = parse_args()
    
    if args.use_biou:
        train_evaluate_value = [["Num", "pixAcc", "unlabel_iou","road_iou","MIoU", "loss", "mBIoU"]]
        val_evaluate_value = [["Num", "pixAcc", "unlabel_iou","road_iou","MIoU", "loss", "mBIoU"]]
    else:
        train_evaluate_value = [["Num", "pixAcc", "unlabel_iou","road_iou","MIoU", "loss"]]
        val_evaluate_value = [["Num", "pixAcc", "unlabel_iou","road_iou","MIoU", "loss"]]

    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        trainer.train()
