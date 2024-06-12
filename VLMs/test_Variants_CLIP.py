###########  Code to test Hugging face multi-modal models  ###
from PIL import Image
import requests, os, json
import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip

img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_names = ['RN50', 'RN101', 'ViT-B-32', 'RN50x4', 'RN50x16', 'ViT-L-14', 'RN50x64', 'roberta-ViT-B-32', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14', 'xlm-roberta-base-ViT-B-32', 'xlm-roberta-large-ViT-H-14', 'ViT-B-16', 'ViT-L-14']
pretrains = ['openai', 'openai', 'openai', 'openai', 'openai', 'openai', 'openai',  'laion2b_s12b_b32k', 'laion2b_s32b_b79k', 'laion2b_s12b_b42k', 'laion2b_s39b_b160k', 'laion5b_s13b_b90k', 'frozen_laion5b_s13b_b90k', 'datacomp_l_s1b_b8k', 'datacomp_xl_s13b_b90k']

fnames = os.listdir(data_path)

#########  Loop over to evaluate each variant of CLIP  ###
for i in range(len(model_names)):
    model_name = model_names[i]
    pretrained = pretrains[i]
    print('=======================================', model_name, '+', pretrained, '=====================', flush = True)
    def load_model(model_name, pretrained, device):
        model, _, transform = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=None,
            device=device)
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, tokenizer, transform

    #######  Define the model, tokenizer and image transform
    model, tokenizer, transform = load_model(model_name, pretrained, device)

    for fname in fnames:  ###  test each subset in the SUGARCREPE++ subset
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

            #########  loop over each sample in the dataset
            for line in data:
                p1 = line['caption']
                neg = line['negative_caption']
                p2 = line['caption2'] # discard = fp[6]
                img_fname = line['filename']
                ipath = os.path.join(img_path, img_fname)
                image = Image.open(ipath)
                model.eval()

                img_feats = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)

                p1_txt = tokenizer(p1).to(device)
                p1_feats = model.encode_text(p1_txt, normalize=True)

                p2_txt = tokenizer(p2).to(device)
                p2_feats = model.encode_text(p2_txt, normalize=True)

                neg_txt = tokenizer(neg).to(device)
                neg_feats = model.encode_text(neg_txt, normalize=True)

                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                cos_p1 = cos(img_feats, p1_feats)  ###  cosine similarities between image and P1 (positive caption 1)
                cos_p2 = cos(img_feats, p2_feats)  ###  cosine similarities between image and P2 (positive caption 2)
                cos_neg = cos(img_feats, neg_feats)  ###  cosine similarities between image and Negative (negative caption)
                cos_p1p2 = cos(p1_feats, p2_feats)  ###  cosine similarities between P1 and P2 for text-only task
                cos_p1_neg = cos(p1_feats, neg_feats)  ###  cosine similarities between P1 and Negative for text-only task
                cos_p2_neg = cos(p2_feats, neg_feats)  ###  cosine similarities between P2 and Negative for text-only task


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
    print('Model evaluation done \n\n')