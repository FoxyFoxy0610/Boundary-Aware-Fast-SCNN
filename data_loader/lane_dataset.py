"""Lane Segmentation Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['LaneSegmentation']


class LaneSegmentation(data.Dataset):
    """Lane Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to dataset folder.
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    dataset_type : string
        'public_cityscapes', 'custom_agricultural', 'asparagus'
    """

    NUM_CLASS = 19 # Default fallback if used without instantiation

    def __init__(self, root='./datasets/citys', split='train', mode=None, transform=None,
                 base_size=520, crop_size=480, dataset_type='custom_agricultural', **kwargs):
        super(LaneSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.dataset_type = dataset_type
        
        self.images, self.mask_paths = _get_path_pairs(self.root, self.split, self.dataset_type)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        
        # --- CLASS MAPPING CONFIGURATION ---
        # Configure class mappings and valid classes based on the dataset_type.
        # This handles the mapping between raw dataset labels and model indices.
        if self.dataset_type == 'custom_agricultural':
            # Tomato class information(unlabel, lane)
            self.num_classes = 2
            LaneSegmentation.NUM_CLASS = 2
            self.valid_classes = [7, 20]
            self._key = np.array([-1, -1, -1, -1, -1, -1,
                                  -1, -1, 0, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1,
                                  -1, -1, -1, 1, -1, -1,
                                  -1, -1, -1, -1, -1, -1,
                                  -1, -1, -1, -1, -1])
        elif self.dataset_type == 'asparagus':
            # Asparagus class information(unlabel, lane, plot)
            self.num_classes = 3
            LaneSegmentation.NUM_CLASS = 3
            self.valid_classes = [7, 20, 25]
            self._key = np.array([-1, -1, -1, -1, -1, -1,
                                  -1, -1, 0, -1, -1, -1,
                                  -1, -1, -1, -1, -1, -1,
                                  -1, -1, -1, 1, -1, -1,
                                  -1, -1, 2, -1, -1, -1,
                                  -1, -1, -1, -1, -1])
        elif self.dataset_type == 'public_cityscapes':
            # Original class information
            self.num_classes = 19
            LaneSegmentation.NUM_CLASS = 19
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33]
            self._key = np.array([-1, -1, -1, -1, -1, -1,
                                  -1, -1, 0, 1, -1, -1,
                                  2, 3, 4, -1, -1, -1,
                                  5, -1, 6, 7, 8, 9,
                                  10, 11, 12, 13, 14, 15,
                                  -1, -1, 16, 17, 18])
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            
            # Fast-SCNN expects exactly (crop_size // 32) * 32
            outsize = (self.crop_size // 32) * 32
            img = img.resize((outsize, outsize), Image.BILINEAR)
            mask = mask.resize((outsize, outsize), Image.NEAREST)
            
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        # Fast-SCNN expects image dimensions to be strictly divisible by 32.
        # We ensure the target dimension is a multiple of 32 by explicit integer division.
        # Because we're in eval, we can just reshape exactly to crop_size, assuming crop_size % 32 == 0
        outsize = (self.crop_size // 32) * 32
        
        # Force exact square of outsize without any aspect ratio preserving to guarantee 
        # that ALL downsample layers cleanly divide by 2 evenly.
        # This completely avoids the (135 != 136) tensor concat errors in feature_fusion.
        img = img.resize((outsize, outsize), Image.BILINEAR)
        mask = mask.resize((outsize, outsize), Image.NEAREST)
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)f
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_path_pairs(folder, split='train', dataset_type='custom_agricultural'):
    def get_paths(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".JPG"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    
                    if dataset_type == 'public_cityscapes':
                        maskname = filename.replace('leftImg8bit', 'gtFine_labelIds').replace('.jpg','.png').replace('.JPG','.png')
                    else:
                        maskname = filename.replace('.png', '_labelIds.png').replace('.jpg', '_labelIds.png').replace('.JPG', '_labelIds.png')
                        
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)

        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        if dataset_type == 'public_cityscapes':
            img_folder = os.path.join(folder, 'leftImg8bit/' + split)
            mask_folder = os.path.join(folder, 'gtFine/' + split)
        else:
            img_folder = os.path.join(folder, 'image/' + split)
            mask_folder = os.path.join(folder, 'label/' + split)
            
        img_paths, mask_paths = get_paths(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        if dataset_type == 'public_cityscapes':
            train_img_folder = os.path.join(folder, 'leftImg8bit/train')
            train_mask_folder = os.path.join(folder, 'gtFine/train')
            val_img_folder = os.path.join(folder, 'leftImg8bit/val')
            val_mask_folder = os.path.join(folder, 'gtFine/val')
        else:
            train_img_folder = os.path.join(folder, 'image/train')
            train_mask_folder = os.path.join(folder, 'label/train')
            val_img_folder = os.path.join(folder, 'image/val')
            val_mask_folder = os.path.join(folder, 'label/val')
            
        train_img_paths, train_mask_paths = get_paths(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_paths(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths
