##########################
from PIL import Image  
import torch, os
import json, sys
import torch.nn as nn
sys.path.append(os.path.join(os.getcwd(), 'CyCLIP/src'))
sys.path.append(os.path.split(os.path.join(os.getcwd(), 'CyCLIP/src'))[0])

current_dir = os.getcwd()
os.chdir(os.path.join(os.getcwd(), 'CyCLIP/src'))
from pkgs.openai.clip import load as load_model
import numpy as np

img_path = os.path.join(current_dir, 'images/') #'path to the images folder'
data_path =  os.path.join(current_dir, 'data/')  #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, processor = load_model(name = 'RN50', pretrained = False)
model = model.to(device)

chkpt = os.path.join(current_dir, 'chkpts/cyclip_3M.pt')
state_dict = torch.load(chkpt, map_location = device)["state_dict"]
if(next(iter(state_dict.items()))[0].startswith("module")):
    state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
fnames = os.listdir(data_path)

def get_inputs(image, caption):
    captions     = processor.process_text(caption)
    pixel_values = processor.process_image(image.convert("RGB"))
    return captions['input_ids'].to(device), captions['attention_mask'].to(device), pixel_values.to(device).unsqueeze(0)
   
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

            in1 = get_inputs(image, p1)
            feats1 = model(input_ids = in1[0], attention_mask = in1[1], pixel_values = in1[2])
            img_feats = feats1.image_embeds
            # img_feats = F.normalize(img_feats,dim=-1)
            p1_feats = feats1.text_embeds

            in2 = get_inputs(image, p2)
            feats2 = model(input_ids = in2[0], attention_mask = in2[1], pixel_values = in2[2])
            p2_feats = feats2.text_embeds

            inneg = get_inputs(image, neg)
            feats_neg = model(input_ids = inneg[0], attention_mask = inneg[1], pixel_values = inneg[2])

            neg_feats = feats_neg.text_embeds

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