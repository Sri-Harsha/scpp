
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch import nn
import torch.optim as optim
import os
import json
from clip.clip import _transform
import numpy as np
import argparse
import sys
from data_organize_internal_data import ImageTextClassificationDataset_sc_mod2
# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def test(args, device, model, test_dataloader):

    total = 0
    total = 0
    correct_img_p1 = 0
    correct_img_p2 = 0
    
    correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
    correct_text = 0
    model.eval()

    for batch in test_dataloader:
        images, tokens_p1, tokens_p2, tokens_neg, img_name  = batch
        images = images.to(device)
        tokens_p1 = tokens_p1.to(device)
        tokens_p2 = tokens_p2.to(device)
        tokens_neg = tokens_neg.to(device)

        with torch.no_grad():
            img_feats, _ = model.encode_image(images)
            txt_p1, _ = model.encode_text(tokens_p1)
            txt_p2, _ = model.encode_text(tokens_p2)
            txt_neg, _ = model.encode_text(tokens_neg)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_p1 = cos(img_feats, txt_p1)
        cos_p2 = cos(img_feats, txt_p2)
        cos_neg = cos(img_feats, txt_neg)
        cos_p1p2 = cos(txt_p1, txt_p2)
        cos_p1_neg = cos(txt_p1, txt_neg)
        cos_p2_neg = cos(txt_p2, txt_neg)

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

def collate_fn_batch(batch):
        cap_p1 = [x[1] for x in batch]
        cap_p2 = [x[2] for x in batch]
        cap_neg = [x[3] for x in batch]
        img_name = [x[4] for x in batch]
        imgs = [x[0] for x in batch]
        imgs = torch.stack(imgs)
        tokens_p1 = clip.tokenize(cap_p1)
        tokens_p2 = clip.tokenize(cap_p2)
        tokens_neg = clip.tokenize(cap_neg)
        return imgs, tokens_p1, tokens_p2, tokens_neg, img_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=20) #400
    parser.add_argument('--mini_batch_size', type=int, default=20)  #400)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--base_model', type=str, default="ViT-B/32")
    parser.add_argument('--betas', type=float, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=10e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--only_evaluate', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--freeze_visual', type=bool, default=True)
    parser.add_argument('--skip_evaluate', type=bool, default=False)
    parser.add_argument('--debugging', type=bool, default=False)
    #--mini_batch_size 20 --batch_size 500 --learning_rate 2e-5
    args = parser.parse_args()

    # assert that batch_size is divisible by mini_batch_size; and that mini_batch_size is =< batch_size
    assert args.batch_size % args.mini_batch_size == 0
    assert args.mini_batch_size <= args.batch_size

    if sys._getframe().f_back:
        args.debugging = True

    torch.manual_seed(args.random_seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device


    # Must set jit=False for training
    model, preprocess = clip.load(args.base_model, device=device, jit=False)

    checkpoint_path = 'chkpts/negCLIP.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['state_dict'])

    if args.freeze_visual:
        for k in model.visual.transformer.parameters():
            k.requires_grad=False


    data_path =  'data/'
    fnames = os.listdir(data_path)


    for fname in fnames:
        print('=======================================================================')
        print('=======================================', fname, '=====================')
        print('=======================================================================')

        test_json_path = os.path.join(data_path, fname)
        test_img_path = 'images/'
        test_dataset = ImageTextClassificationDataset_sc_mod2(test_img_path, test_json_path)
        test_dataloader = DataLoader(test_dataset, collate_fn = collate_fn_batch, batch_size=1)
        
        test(args, device, model, test_dataloader)