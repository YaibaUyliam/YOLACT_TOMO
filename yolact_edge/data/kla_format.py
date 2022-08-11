import os
import time
import json
import glob
import numpy as np
from numpy import random
import torch
from collections import defaultdict
import torch.utils.data as data
import itertools
import cv2 as cv
#from .config import cfg
from pycocotools import mask as maskUtils



# class_label_dictionary = {
#     "5pack": 1
# }

def get_lense_label_map(cfg):
    if cfg.dataset.label_map is None:
        return { x+1 : x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map

def _isArrayLike(obj):
    return (not isinstance(obj, str)) and hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class LENSEAnnotationTransform(object):
    """
        Tra ve bbox [xmin, ymin, xmax, ymax, label_idx]
    """
    def __init__(self,cfg=None):
        if cfg is not None:
            self.dataset_name = cfg.dataset.name
            self.label_map = get_lense_label_map(cfg)

        self.cfg = cfg
    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['classId']
                if label_idx >= 0:
                    label_idx = self.label_map[str(label_idx)] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[2], bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]
            else:
                print("No bbox found for object ", obj)

        return res # [[xmin, ymin, xmax, ymax, label_idx], ... ]

class LENSEDetection(data.Dataset):

    def __init__(self, image_folder, info_folder, transform=None,
                    target_transform=LENSEAnnotationTransform(), has_gt=True, cfg=None):
        print("init len detection")
        self.root = image_folder
        self.chip = LENSE(info_folder, cfg=cfg)
        #self.ids = list(self.chip.imgToAnns.keys())
        self.ids = [i for i in os.listdir(image_folder) if "json" not in i]
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        self.url = image_folder

    def __getitem__(self, index):
        """
            Args:
                index (string): Index
            Returns:
                tuple: Tuple (image, (target, masks))
        """
        #print(f"{index}")
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width)
        """
        # img_id = name of image in annotation file. index = index of list([0, 1, 2, ..., length_dataset - 1])
        img_id = self.ids[index]
        # print('===========')
        # print(img_id)
        #print(f"img_id pullitem: {img_id}")
        #print("img_id: {}".format(img_id))

        if self.has_gt:
            target = self.chip.imgToAnns[img_id]
            ann_ids = self.chip.getAnnIds(imgIds=img_id)
            target = self.chip.loadAnns(ann_ids)
        else:
            target = []

        num_crowds = 0

        image_path = os.path.join(self.url, img_id)
        # print(image_path)
        img = cv.imread(image_path)

        height, width, _ = img.shape

        # masks = [num_masks_image, height, weight]
        if len(target) > 0:
            masks = [self.chip.annToMask(obj, height, width).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4], {'num_crowds': num_crowds, 'labels': target[:, 4]})
                num_crowds = labels['num_crowds']
                labels = labels['labels']
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float32), np.array([[0, 0, 1, 1]]), {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        if target is not None and target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            with open(r"/home/jay2/CONTACT_LENS/Yolact_edge/error.txt", "a+") as errorlog:
                errorlog.write("filename: {}".format(img_id))
            return self.pull_item(random.randint(0, len(self.ids)-1))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, 0


class LENSE:
    def __init__(self, info_folder=None, cfg=None):
        print("LENSE __init__")
        self.dataset = []
        self.anns = dict()
        self.imgToAnns = defaultdict(list)
        self.process_json_folder(info_folder, cfg=cfg)
        self.createIndex()

    def process_json_folder(self, info_folder=None,cfg=None):
        flag_index = 0
        if not info_folder == None:
            print('loading info file into memory...')
            tic = time.time()
            for dir_, _, _ in os.walk(info_folder):
                for json_file in glob.glob(os.path.join(dir_, "*.json")):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        # Trong từng file json.
                        # Chúng ta lặp qua từng defect annotation trong bức ảnh.
                        # Và tạo ra một data_into_dataset sau đó append vào dataset
                        # với key = filename
                        #--------
                        list_classId = data["classId"]
                        for i in range(len(list_classId)):
                            listX = np.array([data["regions"][f"{i}"]["List_X"]], dtype=np.float32).reshape((-1, 1))
                            listY = np.array([data["regions"][f"{i}"]["List_Y"]], dtype=np.float32).reshape((-1, 1))
                            listZ = np.floor(np.concatenate((listX, listY), axis=1))
                            listZ = listZ.astype(np.int32)
                            min_xy = np.min(listZ, axis=0)
                            max_xy = np.max(listZ, axis=0)
                            segmentation = listZ.reshape(1, -1)
                            bbox = np.concatenate((min_xy, max_xy), axis=0)
                            data_into_dataset = {
                                "bbox": bbox.tolist(),
                                "mask": segmentation.tolist(),
                                "classId": cfg.dataset.class_label_dictionary[list_classId[i]],
                                "filename": data["filename"],
                                "id": data["filename"] + str(i)
                            }
                            self.dataset.append(data_into_dataset)
            
            print('Process info from folder success! Time={}'.format(time.time() - tic))
        
    def createIndex(self):
        print("creating index...")
        anns = {}
        imgToAnns = defaultdict(list)
        for ann in self.dataset:
            # Trong imgToAnns chúng ta sẽ lưu trữ:
            # một dictionary với key là tên hình ảnh.
            # giá trị là list các annotation trong hình ảnh đó.
            imgToAnns[ann["filename"]].append(ann)
            anns[ann["id"]] = ann
        
        # anns contains dict{key(filenameimg.json): value:(annotation_value)}
        self.anns = anns
        self.imgToAnns = imgToAnns

    def getAnnIds(self, imgIds = []):

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]

        # if not imgIds need filter then return all annotations
        if len(imgIds) == 0:
            anns = self.dataset
        else:
            # if has imgIds filter
            if not len(imgIds) == 0:
                # then return cac annotations co key == imgId
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset
        
        # ids = list file_name_image linked with annotation
        ids = [ann["id"] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
            return annotations with ids input
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
    
    def annToRLE(self, ann, height, width):
        segm = ann['mask']
        if type(segm) == list:
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = ann['mask']
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m