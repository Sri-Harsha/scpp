#################  Code for ViLT VLM  ###
from transformers import ViltProcessor, ViltModel
import requests, torch
import os, pickle, json
from PIL import Image
import torch.nn as nn
import numpy as np


img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

fnames = os.listdir(data_path)


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
            p2 = line['caption2']
            img_fname = line['filename']
            ipath = os.path.join(img_path, img_fname)
            image = Image.open(ipath).convert("RGB")
            model.eval()

            l1 = len(p1.split(' '))


            if l1 > 25:
                p1 = ' '.join(p1.split()[:25])
                p2 = ' '.join(p2.split()[:25])
                ref = ' '.join(ref.split()[:25])

            encoding = processor(image, p1, return_tensors="pt")
            outputs = model(**encoding)
            p1_outs = outputs.last_hidden_state[:, 0, :]
            p1_outs = nn.functional.normalize(p1_outs, dim=-1)

            encoding = processor(image, p2, return_tensors="pt")
            outputs = model(**encoding)
            p2_outs = outputs.last_hidden_state[:, 0, :]
            p2_outs = nn.functional.normalize(p2_outs, dim=-1)

            encoding = processor(image, ref, return_tensors="pt")
            outputs = model(**encoding)
            neg_outs = outputs.last_hidden_state[:, 0, :]
            neg_outs = nn.functional.normalize(neg_outs, dim=-1)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            p1_neg = cos(p1_outs, neg_outs)
            p2_neg = cos(p2_outs, neg_outs)
            p1_p2 = cos(p1_outs, p2_outs)
            total += 1
            
            if p1_p2 > p1_neg and p1_p2 > p2_neg:# and cos_refd < cos_p1 and cos_refd < cos_p2 and cos_refd < cos_ref:
                correct_full += 1
            if p1_p2 > p1_neg:# and cos_refd < cos_p1 and cos_refd < cos_p2 and cos_refd < cos_ref:
                correct_img_p1 += 1
            if p1_p2 > p2_neg:# and cos_refd < cos_p1 and cos_refd < cos_p2 and cos_refd < cos_ref:
                correct_img_p2 += 1
            

        print(f"====== evaluation results ======", flush = True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score}", flush = True)

        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1}", flush = True)

        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2}", flush = True)


        # print('Do not have text-only results')