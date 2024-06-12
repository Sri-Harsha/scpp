###########  Code to test Teaching Structured Vision & Language Concepts model  ###
from PIL import Image  
from torch import nn
import numpy as np
import os, torch, sys, json
sys.path.append(os.path.join(os.getcwd(), 'Structured_VL/src'))
sys.path.append(os.path.split(os.path.join(os.getcwd(), 'Structured_VL/src'))[0])

current_dir = os.getcwd()
os.chdir(os.path.join(os.getcwd(), 'Structured_VL/src'))

import open_clip as clip
from open_clip import create_model_and_transforms, trace_model
from training.params import parse_args
from training.distributed import is_master, init_distributed_device, world_info_from_env
args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_path = os.path.join(current_dir, 'images/') #'path to the images folder'
data_path =  os.path.join(current_dir, 'data/')  #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
chkpt_path = os.path.join(current_dir, 'chkpts/cc3m_llm_and_rb_negs_epoch_5.pt')


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

args.model = args.model.replace('/', '-')
args.lora = 4
args.pretrained = 'openai'

model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        lora = args.lora,
        freeze_img = args.freeze_img)

checkpoint = torch.load(chkpt_path, map_location='cpu')
sd = checkpoint["state_dict"]
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in sd.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

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
            image = preprocess_val(Image.open(ipath)).unsqueeze(0).to(device)
            model.eval()
    
            p1_txt = clip.tokenize(p1).to(device)
            img_feats, p1_feats, logit_scale = model(image, p1_txt)
            p2_txt = clip.tokenize(p2).to(device)
            img_feats, p2_feats, logit_scale = model(image, p2_txt)
            
            neg_txt = clip.tokenize(neg).to(device)
            img_feats, neg_feats, logit_scale = model(image, neg_txt)

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