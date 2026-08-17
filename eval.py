import os
import csv
import torch
import torch.utils.data as data
from torchvision.transforms import v2

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete
from utils.loss import MixSoftmaxCrossEntropyOHEMLoss, BoundaryAwareLoss

from train import parse_args

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        # Ensure output folder for masks exists
        self.outdir = 'pred_result'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Image transforms for evaluation
        input_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Dataset and dataloader
        data_kwargs = {
            'root': args.data_path,
            'transform': input_transform, 
            'base_size': args.base_size, 
            'crop_size': args.crop_size,
            'dataset_type': args.dataset_type
        }

        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', **data_kwargs)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
        
        # Load network
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=True, weight_path=args.weight_path).to(args.device)
        print('Finished loading model!')

        self.metric = SegmentationMetric(val_dataset.num_class)

        # Initialize Base Criterion and BoundaryAwareLoss to always calculate BIoU metric
        base_criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight, ignore_index=-1)
        # Always instantiate BoundaryAwareLoss to calculate BIoU for metrics, regardless of use_biou config
        self.criterion = BoundaryAwareLoss(base_loss_fn=base_criterion, dilation_ratio=args.dilation_ratio).to(args.device)

    def eval(self):
        self.model.eval()
        
        # Setup CSV export for image-level evaluation
        csv_filename = os.path.join(self.outdir, 'evaluation_metrics.csv')
        csv_data = []
        
        # Headers: We don't know the exact number of classes until we run, but let's assume based on dataset_type
        # We will add headers dynamically or just define general ones.
        
        sample = 0
        mean_pixAcc = 0.0
        mean_MIoU = 0.0
        mean_mBIoU = 0.0

        with torch.no_grad():
            for i, (image, label) in enumerate(self.val_loader):
                # Note: To get filename, lane_dataset.py mode='testval' must return filename if possible.
                # Actually, lane_dataset.py returns `img, mask` in 'testval' mode. 
                # Let's just use index as filename if filename is not returned.
                image = image.to(self.args.device)
                label = label.to(self.args.device)

                # Forward pass
                outputs = self.model(image)
                
                # Predict class indices
                pred = torch.argmax(outputs[0], 1)
                pred_numpy = pred.cpu().data.numpy()
                label_numpy = label.cpu().data.numpy()

                # Calculate standard metrics (pixAcc, MIoU, IoU per class)
                self.metric.reset() # Reset metric per image to get individual image metric
                self.metric.update(pred_numpy, label_numpy)
                pixAcc, MIoU, IoU = self.metric.get()

                # Calculate BIoU
                # We use biou_ratio=0.0 so we only compute the BIoU metric without affecting the loss value
                loss, mbiou = self.criterion(outputs, label, biou_ratio=0.0)

                print('Sample %d, pixAcc: %.3f%%, MIoU: %.3f%%, BIoU: %.3f%%' % (i + 1, pixAcc * 100, MIoU * 100, mbiou * 100))

                # Prepare row data
                row = [f"Sample_{i+1}", pixAcc, MIoU, float(mbiou)] + list(IoU)
                csv_data.append(row)

                mean_pixAcc += pixAcc
                mean_MIoU += MIoU
                mean_mBIoU += float(mbiou)
                sample += 1

                # Save predicted mask visualization
                predict = pred.squeeze(0).cpu().data.numpy()
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(self.outdir, f'seg_{i+1}.png'))

        # Calculate means
        if sample > 0:
            mean_pixAcc /= sample
            mean_MIoU /= sample
            mean_mBIoU /= sample
            
        print('=====================================')
        print('Mean pixAcc: %.4f || Mean MIoU: %.4f || Mean BIoU: %.4f' % (mean_pixAcc, mean_MIoU, mean_mBIoU))
        
        # Save to CSV
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Create header
            num_classes = len(csv_data[0]) - 4 # Total length - (Name, pixAcc, MIoU, BIoU)
            header = ["Image_Name", "pixAcc", "MIoU", "mBIoU"] + [f"IoU_Class_{c}" for c in range(num_classes)]
            writer.writerow(header)
            writer.writerows(csv_data)
            
        print(f"Evaluation metrics saved to: {csv_filename}")

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print('Testing model: ', args.model)
    evaluator.eval()
