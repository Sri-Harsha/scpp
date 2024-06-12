###########  Code to test Hugging face multi-modal models  ###
from PIL import Image
import requests, os, torch, pickle, json
import torch.nn as nn
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor
import numpy as np


img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FlavaForPreTraining.from_pretrained("facebook/flava-full").eval().to(device)
feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
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

            p1_input = tokenizer(text=p1, return_tensors="pt", padding="max_length", max_length=77).to(device)
            p1_feats = model.flava.get_text_features(**p1_input)[:, 0, :]  #.cpu().detach().numpy()

            p2_input = tokenizer(text=p2, return_tensors="pt", padding="max_length", max_length=77).to(device)
            p2_feats = model.flava.get_text_features(**p2_input)[:, 0, :]

            neg_input = tokenizer(text=ref, return_tensors="pt", padding="max_length", max_length=77).to(device)
            neg_feats = model.flava.get_text_features(**neg_input)[:, 0, :]

            #############  image  features  ####

            img_input = feature_extractor(images=image, return_tensors="pt").to(device)
            img_feats = model.flava.get_image_features(**img_input)[:, 0, :]


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