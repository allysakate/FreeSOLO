"""
https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py

Balloon Sample:
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=UkNbUzUOLYf0
balloon sample: {
    "34020010494_e5cb88e1c4_k.jpg1115004":{
        "fileref":"","size":1115004,
        "filename":"34020010494_e5cb88e1c4_k.jpg",
        "base64_img_data":"",
        "file_attributes":{},
        "regions":{"0":{
            "shape_attributes":{
                "name":"polygon",
                "all_points_x":[1020,1000,994,1003,1023,1050, ...],
                "all_points_y":[963,899,841,787,738,700,663, ...]
                },
            "region_attributes":{}
            }
        }
        },

MOTS: https://www.vision.rwth-aachen.de/page/mots
class_id: car 1, pedestrian 2
format: time_frame id class_id img_height img_width rle
mots sample: 1 2002 2 1080 1920 UkU\1`0RQ1>PoN\OVP1X1F=I3oSOTNlg0U2l ...

COCO Format:
{
    "categories": [
        {
            "supercategory": "person", 
            "id": 1, 
            "name": "person"
        }, ...
    ], 
    "images": [
        {
            "file_name": "train2017/000000516808.jpg", 
            "height": 425, 
            "width": 640, 
            "id": 516808
        }, ...
    ], 
    "annotations": [
        {
            "segmentation": {
                "size": [425, 640], 
                "counts": "[W^3o0Q<;G9H8 ..."
            },
            "bbox": [left, top, width, height,
            "score": 0.7974770069122314, 
            "iscrowd": 0, 
            "image_id": 516808,
            "category_id": 1,
            "id": 1
        }, ...
    ]
}

"""
import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import torch
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks, save_image
from torchvision.io import read_image
import os
import cv2
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import shutil

class SegmentedObject:
    def __init__(self, mask, class_id, track_id, bbox, mask_bool):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id
        self.bbox = bbox
        self.mask_bool = mask_bool

def load_image(mots_dir, group, filename, is_visualize= False, id_divisor=1000):
    img = np.array(Image.open(f"{mots_dir}/instances/{group}/{filename}.png"))
    obj_ids = np.unique(img)
    objects = []
    mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
    for idx, obj_id in enumerate(obj_ids):
        if obj_id == 0:  # background
            continue

        mask.fill(0)
        pixels_of_elem = np.where(img == obj_id)
        mask[pixels_of_elem] = 1
        mask_bool = torch.from_numpy(mask.astype(bool))
        mask_t = torch.from_numpy(mask)
        mask_torch = mask_t.unsqueeze(0)
        bbox = masks_to_boxes(mask_torch).tolist()[0]
        bbox = list(map(int, masks_to_boxes(mask_torch)[0]))
        seg_obj = SegmentedObject(
            rletools.encode(mask),
            obj_id // id_divisor,
            obj_id,
            bbox,
            mask_bool
        )
        objects.append(seg_obj)
    return objects


def get_mots(data_split):
    # https://www.vision.rwth-aachen.de/page/mots
    # class_id: car 1, pedestrian 2
    # format: time_frame id class_id img_height img_width rle

    # balloon sample: {"34020010494_e5cb88e1c4_k.jpg1115004":{"fileref":"","size":1115004,"filename":"34020010494_e5cb88e1c4_k.jpg","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[1020,1000,994,1003,1023,1050,1089,1134,1190,1265,1321,1361,1403,1428,1442,1445,1441,1427,1400,1361,1316,1269,1228,1198,1207,1210,1190,1177,1172,1174,1170,1153,1127,1104,1061,1032,1020],"all_points_y":[963,899,841,787,738,700,663,638,621,619,643,672,720,765,800,860,896,942,990,1035,1079,1112,1129,1134,1144,1153,1166,1166,1150,1136,1129,1122,1112,1084,1037,989,963]},"region_attributes":{}}}},"25899693952_7c8b8b9edc_k.jpg814535":{"fileref":"","size":814535,"filename":"25899693952_7c8b8b9edc_k.jpg","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[586,510,441,389,331,268,196,158,135,156,210,307,403,437,443,425,448,481,506,513,503,511,502,583,669,755,800,811,803,784,755,717,668,620,586],"all_points_y":[133,116,115,126,149,185,261,339,438,560,664,784,868,893,929,964,964,961,965,958,936,910,896,836,752,636,538,454,392,329,270,220,175,149,133]},"region_attributes":{}}}}

    # mots sample: 1 2002 2 1080 1920 UkU\1`0RQ1>PoN\OVP1X1F=I3oSOTNlg0U2lWOVNng0m1nWOWNlg0n1PXOWNlg0l1SXOUNjg0P2RXORNfg0V2WXOoMbg0V2\XOlM^g0Z2^XOkM_g0W2^XOlM]g0Y2`XOjM]g0Y2`XOkM[g0Y2aXOkM^g0V2`XOlM^g0T2aXOoM\g0m1hXOVNWg0^1eVOeLU2o1Ug0P1`YORO^f0n0cYOTO[f0j0eYOYOZf0e0hYO\OWf0=oYOEoe04XZONge00[ZO3be0K`WOnM@HY2a2ff0J]WOXNk1P2ff0J\WOYNk1P2if0GXWO]Nm1n1jf0GTWObNm1j1nf0ERWOdNn1h1of0o0kXOUOUg0m0fXOVOYg0m0cXOUO]g0o0\XOTObg0o0\XOROdg0Q1WXOROgg0P1UXOSOjg0nNXWOBh0e0[OUNeh0c1YWOEd0c0CQN_h0h1ZWOEb0b0GPN[h0k1\WOE=a00nMVh0l1[WOI<=4nMTh0Y7oWOeHQh0f1]WO[3b0oJPh0g1aWOT3c0VKjg0f1eWOQ3b0YKhg0g1fWOn2e0ZKeg0h1dWOn2i0ZKcg0h1cWOo2k0XKag0j1cWOm2n0YK_g0j1bWOn2o0XK_g0j1bWOj2U1ZKWg0n1dWOf2W1\KTg0o1eWOb2Z1_KQg0o1eWOa2[1`Kof0P2fWO]2^1cKkf0Q2gWO[2_1dKjf0P2hWOZ2`1fKff0R2jWOT2e1gKbf0V2iWOP2i1iK\f0Z2jWOe1ni0PLYVOj6ai0h0O3O1O100N4N[K`VOT1]i0lNdVOV1Zi0iNgVOY1Wi0dNkVO_1Si0aNkVOd1Qi0\NoVOh1nh0XNQWOk1mh0UNRWOm1mh0c3L1O2N3L5L1O2N2N1O4L1O4L10iHUXO`4jg0[K[XOf4dg0WK_XOk4`g0UJ]XOWO3T33A\g0RNaXOVO0S3;CUg0RNaXOo2e0mNgf0TNdXOj2n0oN^f0UNeXOb2\1VOne0WNgXOW1@nNP2d1ke0QNgXOW1JjNm1k1de0oMgXO[1J^NW2X2_e0cMdXOf1KVNZ2^2[f0\OfZOa0Ye0]OkZOa0Se0@S[Ob0dd0^O`[Oa0]d0@k[O:Rd0Ho[O5Rd0Mo[O2oc0OP\O1oc01Q\ONQd03n[OMPd03Q\OLoc06P\OMlc03T\OMmc02U\OMjc03V\OLlc05R\OKPd03Q\OKQd04o[OLWd00h[ONZd01h[OL^d0Oe[OM^d00c[OOad0O_[OOdd0N^[O0fd0MZ[O2jd0JZ[O2hd0MY[O1kd0LV[O2kd0MW[O1md0LS[O2od0MR[O2Qe0NmZO0Ue00jZONYe01iZOMZe03dZOK]e05dZOI_e07bZOF_e09bZOEbe09^ZOFbe0<\ZODfe0;YZOBme0<RZODoe0<PZOCUf0=gYOCZf0?cYOA`f0>^YOAdf0?[YOAff0a0WYO]Olf0c0SYO]OPg0d0mXO[OTg0h0hXOWOZg0i0eXOWO_g0f0`XOYObg0f0`XOXOdg0d0\XO\Oeg0c0[XO\Ogg0d0YXO[Ogg0e0ZXOXOjg0h0c4O2N2L4M7E^l`=
    folder_path = "/media/catchall/starplan/Dissertation/Dataset/MOTSChallenge/train/"
    text_file = os.path.join(folder_path, f"{data_split}.txt")

    # Check whether the specified path exists or not
    images_path = os.path.join(folder_path, f"mots_{data_split}")
    if not os.path.exists(images_path):
        # Create a new directory because it does not exist 
        os.makedirs(images_path)
        print(f"Directory {images_path} is created!")

    data_dict = {}
    coco_dict = {}
    coco_dict["categories"] = [
        {
            "supercategory": "vehicle", 
            "id": 1, 
            "name": "car"
        },
        {
            "supercategory": "person", 
            "id": 2, 
            "name": "person"
        }
    ]
    with open(text_file) as f:
        for line in f:
            fields = line.strip().split(" ")
            try:
                group = int(fields[0])
                image_id = int(fields[1]) # 2 - image basename
                class_id = int(fields[4]) # 3 - frame id
                height = int(fields[5])
                width = int(fields[6])
                bbox_int = list(map(int, fields[7:11]))
                rle = {"size": [height, width], "counts": fields[11].encode(encoding="UTF-8")}
                if image_id in data_dict.keys():
                    data_dict[image_id]["annotations"].append({
                        "bbox": bbox_int,
                        "segmentation": rle,
                        "category_id": class_id
                    })
                else:
                    orig_image = f"{folder_path}new_images/{fields[0]}/{fields[2]}"
                    new_image = os.path.join(images_path, f"{fields[0]}_{fields[2]}")
                    shutil.copy(orig_image, new_image)
                    data_dict[image_id] = {}
                    data_dict[image_id]["file_name"] = new_image
                    data_dict[image_id]["image_id"] = image_id
                    data_dict[image_id]["height"] = height
                    data_dict[image_id]["width"] = width
                    data_dict[image_id]["annotations"] = [
                        {   
                            "bbox": bbox_int,
                            "segmentation": rle,
                            "category_id": class_id
                        }
                    ]
            except IndexError as e:
                print(fields)
        image_list = []
        ann_list = []
        ann_cnt = 1
        for image_id, value in data_dict.items():
            image_data = {
                "file_name": value["file_name"], 
                "height": value["height"], 
                "width": value["width"], 
                "id": image_id
            }
            image_list.append(image_data)
            for ann in value["annotations"]:
                rle = ann["segmentation"]
                ann_data = {
                    "segmentation": {
                        "size": rle["size"],
                        "counts": rle["counts"].decode(encoding='UTF-8')
                    },
                    "bbox": ann["bbox"],
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "id": ann_cnt
                }
                ann_list.append(ann_data)
                ann_cnt += 1
        coco_dict["images"] = image_list
        coco_dict["annotations"] = ann_list
    return coco_dict, ann_cnt

if __name__ == "__main__":
    data_split = "train"
    coco_dict, ann_cnt = get_mots(data_split)
    print(f"last annotation id: {ann_cnt}")
    with open(f"{data_split}.json", "w") as outfile:
        json.dump(coco_dict, outfile)

    with open(f"{data_split}.json", "r") as f:
        data = json.load(f)
    print(data)
    print(f"type: {type(data)}")
    print(f"length: {len(data)}")
    print(f"keys: {data.keys()}")