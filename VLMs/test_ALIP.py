import sys, os, torch, pickle, json
sys.path.append(os.path.join(os.getcwd(), 'ALIP'))  ######  provide path to the ALIP folder
from utils import get_state_dict, get_transform
import torch.nn as nn
from src.open_alip import create_model, tokenize
from PIL import Image
import torch.nn.functional as F
import numpy as np

img_path = 'images/' #'path to the images folder'
data_path =  'data/' #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = get_transform(224)  #image_size = 224
chkpt_path = 'chkpts/ALIP_YFCC15M_B32.pt'  ####  
tokenizer = tokenize

model = create_model("ViT-B/32")
state_dict = get_state_dict(chkpt_path)
model.load_state_dict(state_dict, strict=True)
model.cuda()


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

            image = transform(image).unsqueeze(axis = 0)
            img_feats = model.encode_image(image.cuda())
            img_feats = F.normalize(img_feats,dim=-1)

            toks_p1 = tokenizer(p1).to(device)
            p1_feats =  model.encode_text(toks_p1)
            p1_feats = F.normalize(p1_feats,dim=-1)   

            toks_p2 = tokenizer(p2).to(device)
            p2_feats =  model.encode_text(toks_p2)
            p2_feats = F.normalize(p2_feats,dim=-1)

            toks_neg = tokenizer(neg).to(device)
            neg_feats =  model.encode_text(toks_neg)
            neg_feats = F.normalize(neg_feats,dim=-1)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_p1 = cos(img_feats, p1_feats)
            cos_p2 = cos(img_feats, p2_feats)
            cos_neg = cos(img_feats, neg_feats)
            cos_p1p2 = cos(p1_feats, p2_feats)
            cos_p1_neg = cos(p1_feats, neg_feats)
            cos_p2_neg = cos(p2_feats, neg_feats)

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