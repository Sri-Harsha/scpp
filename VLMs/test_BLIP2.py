#####################################
from PIL import Image
import torch, os
import pickle, json
import torch.nn as nn
from lavis.models import load_model_and_preprocess
import numpy as np


img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
image_size = 384
device = 'cpu'  #'cuda' if torch.cuda.is_available() else 'cpu'
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

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
            neg = line['negative_caption']
            p2 = line['caption2']
            img_fname = line['filename']
            ipath = os.path.join(img_path, img_fname)
            image = Image.open(ipath).convert("RGB")
            model.eval()

            img_in = vis_processors["eval"](image).unsqueeze(0).to(device)
            p1_in = txt_processors["eval"](p1)
            sample_p1 = {"image": img_in, "text_input": [p1_in]}
            img_feats = model.extract_features(sample_p1, mode="image")
            img_feats = img_feats.image_embeds_proj[:, 0, :]

            p1_feats = model.extract_features(sample_p1, mode="text")
            p1_feats = p1_feats.text_embeds_proj[:,0,:]

            p2_in = txt_processors["eval"](p2)
            sample_p2 = {"image": img_in, "text_input": [p2_in]}
            p2_feats = model.extract_features(sample_p2, mode="text")
            p2_feats = p2_feats.text_embeds_proj[:,0,:]

            neg_in = txt_processors["eval"](neg)
            sample_neg = {"image": img_in, "text_input": [neg_in]}
            neg_feats = model.extract_features(sample_neg, mode="text")
            neg_feats = neg_feats.text_embeds_proj[:,0,:]

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(img_feats, p1_feats)
            cos_p2 = cos(img_feats, p2_feats)
            cos_neg = cos(img_feats, neg_feats)
            cos_p1p2 = cos(p1_feats, p2_feats)
            cos_p1_neg = cos(p1_feats, neg_feats)
            cos_p2_neg = cos(p2_feats, neg_feats)

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