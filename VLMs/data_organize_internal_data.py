import os
# import cv2
from clip.clip import _transform
import json
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import torch
from torch.utils.data import Dataset
process = _transform(224)


# load image in real time version
class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, csv_path):  #, model_type="clip", vilt_processor=None):
        self.img_path = img_path
        # self.model_type = model_type
       
        with open(csv_path, "r") as f:
            lines = f.readlines()
        req_lines = []
        for line in lines:
            fp = line.split('\t')
            if fp[6] == '' and fp[7] == '':
                continue
            else:
                req_lines.append(line)
        self.req_lines = req_lines[1:]
            
    def __getitem__(self, idx):
        line = self.req_lines[idx]
        fp = line.split('\t')
        
        p1 = fp[4]
        p2 = fp[5]
        ref = fp[6]
        img_name = fp[0]
        ipath = os.path.join(self.img_path, img_name)
        image = process(Image.open(ipath))

        return image, p1, p2, ref, img_name
    
    def __len__(self):
        return len(self.req_lines)

class ImageTextClassificationDataset_sc_mod2(Dataset):
    # load image in real time version
    def __init__(self, img_path, json_path):  #, model_type="clip", vilt_processor=None):

        self.img_path = img_path
        # self.model_type = model_type
        f = open(json_path)
        #self.json_path = json_path

        self.data = json.load(f)
    
    def __getitem__(self, idx):

        line = self.data[idx]
        
        p1 = line['caption']
        ref = line['negative_caption']
        p2 = line['caption2'] # discard = fp[6]
        img_fname = line['filename']
        ipath = os.path.join(self.img_path, img_fname)
        image = process(Image.open(ipath))
        #img_name = fp[0]
        return image, p1, p2, ref, img_fname
    
    def __len__(self):
        
        nlines = len(self.data)
        return nlines

class ImageTextClassificationDataset_sc_mod(Dataset):
    # load image in real time version
    def __init__(self, img_path, csv_path):  #, model_type="clip", vilt_processor=None):
        self.img_path = img_path
        # self.model_type = model_type
        with open(csv_path, "r") as f:
            lines = f.readlines()
        
        self.req_lines = lines[1:]
            
    def __getitem__(self, idx):
        line = self.req_lines[idx]
        fp = line.split('\t')
        ipath = os.path.join(self.img_path, fp[0])
        p1 = fp[1]
        ref = fp[3]
        p2 = fp[2] # discard = fp[6]
        image = process(Image.open(ipath))
        img_name = fp[0]
        return image, p1, p2, ref, img_name
    
    def __len__(self):
        return len(self.req_lines)


class ImageTextClass_data(Dataset):
    def __init__(self, img_path, csv_path):  #, model_type="clip", vilt_processor=None):
        self.img_path = img_path
        # self.model_type = model_type
       
        with open(csv_path, "r") as f:
            lines = f.readlines()
        self.req_lines = lines[1:]
            
    def __getitem__(self, idx):

        line = self.req_lines[idx]
        fp = line.split('\t')

        p1 = fp[1]
        p2 = fp[2]
        ref = fp[3]

        img_name = fp[0]
        ipath = os.path.join(self.img_path, img_name)
        image = process(Image.open(ipath))

        return image, p1, p2, ref, img_name
    
    def __len__(self):
        
        #return len(self.images)
        return len(self.req_lines)