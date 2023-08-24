import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
ROOT_DIR = "D:\mcn"
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "log")
class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 5
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "Healthy")
        self.add_class("object", 2, "Mosaic")
        self.add_class("object", 3, "RedRot")
        self.add_class("object", 4, "Rust")
        self.add_class("object", 5, "Yellow")
        assert subset in ["trainS", "valS"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations_file = os.path.join(dataset_dir, f"{subset}.json")
        with open(annotations_file, "r") as f:
            annotations1 = json.load(f)
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['type'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"Healthy": 1,"Mosaic": 2,"RedRot":3,"Rust":4,"Yellow":5}
            num_ids = [name_dict[a] for a in objects]
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:\mcn", "trainS")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("D:\mcn", "valS")
    dataset_val.prepare()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
train(model)