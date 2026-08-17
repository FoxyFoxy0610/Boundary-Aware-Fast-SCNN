import os
import argparse
import torch
from PIL import Image
from torchvision.transforms import v2

from models.fast_scnn import get_fast_scnn
from utils.visualize import get_color_pallete

def parse_args():
    """Parse prediction arguments."""
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (e.g. citys)')
    parser.add_argument('--dataset-type', type=str, default='custom_agricultural',
                        help='dataset type (e.g. custom_agricultural, citys)')
    parser.add_argument('--weight-path', default='./weights/weight_demo.pth',
                        help='Path to checkpoint model (.pth file)')
    parser.add_argument('--input-pic', type=str, default='./datasets/test_lane.png',
                        help='path to the input picture')
    parser.add_argument('--outdir', default='./pred_result', type=str,
                        help='path to save the predict result')
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='use CPU for prediction')
    parser.set_defaults(cpu=False)
    return parser.parse_args()

def predict():
    """Main prediction workflow for a single image."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Ensure output folder exists
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Basic image transform to tensor
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    # Load input image
    if not os.path.isfile(args.input_pic):
        print(f"Error: Input picture not found at {args.input_pic}")
        return

    print(f"Loading image from: {args.input_pic}")
    image = Image.open(args.input_pic).convert('RGB')
    
    # Fast-SCNN expects dimensions exactly divisible by 32
    w, h = image.size
    out_w = (w // 32) * 32
    out_h = (h // 32) * 32
    image = image.resize((out_w, out_h), Image.BILINEAR)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Force the class count in LaneSegmentation to match our dataset-type before model init
    from data_loader import datasets
    if args.dataset_type == 'custom_agricultural':
        datasets[args.dataset].NUM_CLASS = 2
    elif args.dataset_type == 'asparagus':
        datasets[args.dataset].NUM_CLASS = 3
        
    # Load model
    model = get_fast_scnn(args.dataset, pretrained=True, weight_path=args.weight_path, map_cpu=args.cpu, dataset_type=args.dataset_type).to(device)
    print('Finished loading model!')
    model.eval()
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Get mask prediction (extract argmax for class indices)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    
    # Convert prediction to color mask using the palette defined for the dataset
    mask = get_color_pallete(pred, args.dataset)
    
    # Save the predicted color mask
    outname = os.path.splitext(os.path.basename(args.input_pic))[0] + '_mask.png'
    outpath = os.path.join(args.outdir, outname)
    mask.save(outpath)
    print(f"Mask saved to: {outpath}")

if __name__ == '__main__':
    predict()
