from .lane_dataset import LaneSegmentation

datasets = {
    'lane': LaneSegmentation,
    'citys': LaneSegmentation, # Keep citys for backward compatibility if dataset=citys is passed
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
