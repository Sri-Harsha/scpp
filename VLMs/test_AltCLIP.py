###########  Code to test Hugging face multi-modal models  ###
from PIL import Image
import torch, pickle, json, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, AltCLIPModel

img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
image_size = 384
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fnames = os.listdir(data_path)

model = AltCLIPModel.from_pretrained("BAAI/AltCLIP").to(device)
processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

for fname in fnames:
        print('=======================================================================', flush = True)
        print('=======================================', fname, '=====================', flush = True)
        json_path = os.path.join(data_path, fname)
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0

        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        correct_text = 0

        f = open(json_path)
        data = json.load(f)


        for line in data:
            p1 = line['caption']
            ref = line['negative_caption']
            p2 = line['caption2'] # discard = fp[6]
            img_fname = line['filename']
            ipath = os.path.join(img_path, img_fname)
            image = Image.open(ipath).convert("RGB")
            model.eval()

            inputs = processor(images=image, return_tensors="pt").to(device)

            img_feats = model.get_image_features(**inputs)
            img_feats = F.normalize(img_feats,dim=-1)
            
            inputs1 = processor(text=p1, padding=True, return_tensors="pt").to(device)
            p1_feats = model.get_text_features(**inputs1)
            p1_feats = F.normalize(p1_feats,dim=-1)   

            inputs2 = processor(text=p2, padding=True, return_tensors="pt").to(device)
            p2_feats = model.get_text_features(**inputs2)
            p2_feats = F.normalize(p2_feats,dim=-1)

            inputsr = processor(text=ref, padding=True, return_tensors="pt").to(device)
            neg_feats = model.get_text_features(**inputsr)
            neg_feats = F.normalize(neg_feats,dim=-1) 

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(img_feats, p1_feats)  ###  cosine similarities between image and P1 (positive caption 1)
            cos_p2 = cos(img_feats, p2_feats)  ###  cosine similarities between image and P2 (positive caption 2)
            cos_neg = cos(img_feats, neg_feats)  ###  cosine similarities between image and Negative (negative caption)
            cos_p1p2 = cos(p1_feats, p2_feats)  ###  cosine similarities between P1 and P2 for text-only task
            cos_p1_neg = cos(p1_feats, neg_feats)  ###  cosine similarities between P1 and Negative for text-only task
            cos_p2_neg = cos(p2_feats, neg_feats)  ###  cosine similarities between P2 and Negative for text-only task

            #############  Compute the performance of the models on each subset  ###

            total += 1

            if cos_p1 > cos_neg and cos_p2 > cos_neg:
                correct_full += 1
            if cos_p1 > cos_neg:
                correct_img_p1 += 1
            if cos_p2 > cos_neg:
                correct_img_p2 += 1
            if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg:
                correct_text += 1
        
        print(f"====== evaluation results ======", flush = True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score}", flush = True)
        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1}", flush = True)
        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2}", flush = True)


        ave_score_txt = float(correct_text) / float(total)
        print(f"Accuracy text-only task: {ave_score_txt}", flush = True)