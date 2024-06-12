###############  ViLT trained for ITR   ###
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import torch
import os, pickle, json
from PIL import Image
# import torch.nn as nn
import numpy as np


img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

fnames = os.listdir(data_path)


for fname in fnames:
        print('=======================================================================', flush = True)
        print('=======================================', fname, '=====================', flush = True)
        json_path = os.path.join(data_path, fname)
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0

        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        #correct_text = 0

        f = open(json_path)
        data = json.load(f)


        for line in data:
            p1 = line['caption']
            neg = line['negative_caption']
            p2 = line['caption2']
            img_fname = line['filename']
            ipath = os.path.join(img_path, img_fname)
            image = Image.open(ipath).convert("RGB")
            model.eval()

            l1 = len(p1.split(' '))
            if l1 > 25:
                p1 = ' '.join(p1.split()[:25])
                p2 = ' '.join(p2.split()[:25])
                neg = ' '.join(neg.split()[:25])
            

            encoding = processor(image, p1, return_tensors="pt")
            outputs = model(**encoding)
            sc_p1 = outputs.logits[0, :].item()

            encoding = processor(image, p2, return_tensors="pt")
            outputs = model(**encoding)
            sc_p2 = outputs.logits[0, :].item()


            encoding = processor(image, neg, return_tensors="pt")
            outputs = model(**encoding)
            sc_neg = outputs.logits[0, :].item()
            

            total += 1

            if sc_p1 > sc_neg and sc_p2 > sc_neg:
                correct_full += 1
            if sc_p1 > sc_neg:
                correct_img_p1 += 1
            if sc_p2 > sc_neg:
                correct_img_p2 += 1

        print(f"====== evaluation results ======", flush = True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score}", flush = True)

        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1}", flush = True)


        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2}", flush = True)