###########  Code to test Hugging face multi-modal models  ###
import torch 
import os, pickle, json, argparse, sys
from PIL import Image
image_size = 224
sys.path.append(os.path.join(os.getcwd(), 'SGVL/BLIP'))
sys.path.append(os.path.split(os.path.join(os.getcwd(), 'SGVL/BLIP'))[0])

current_dir = os.getcwd()
os.chdir(os.path.join(os.getcwd(), 'SGVL/BLIP'))
from Winoground.evaluate_winoground import blip_processor
from models.blip_retrieval_vg import blip_retrieval_vg
import ruamel.yaml as yaml
import numpy as np

img_path = os.path.join(current_dir, 'images/') #'path to the images folder'
data_path =  os.path.join(current_dir, 'data/')  #'path to folder with caption files'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
chkpt = os.path.join(current_dir, 'chkpts/BLIP_SGVL.pt')

parser = argparse.ArgumentParser()     
parser.add_argument('--config', default='./configs/laion_vg.yaml')
parser.add_argument('--output_dir', default='output')
parser.add_argument("--name", default="test")
parser.add_argument('--evaluate', default = chkpt, type=str)        
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)
parser.add_argument('--train-data', default = None, type=str)
parser.add_argument('--train-num-samples', default = 0, type=int)
parser.add_argument('--dataset-type', default = "auto", type=str)
parser.add_argument('--workers', default = 4, type=int)
parser.add_argument('--vg-data', default = "../Data", type=str)
parser.add_argument('--vg-loss-lambda', default = 1.0, type=float)
parser.add_argument('--negatives-loss-lambda', default = 1.0, type=float)
parser.add_argument('--negatives', action='store_true')
parser.add_argument('--batch-size', default = 32, type=int)
parser.add_argument('--vg-batch-size', default = 8, type=int)
parser.add_argument('--objects', default = 10, type=int)
parser.add_argument('--object-tokens', default = 25, type=int)
parser.add_argument('--relations', default = 7, type=int)
parser.add_argument('--relation-tokens', default = 7, type=int)
parser.add_argument('--head-layers', default = 3, type=int)
parser.add_argument('--winoground', action='store_true')
parser.add_argument('--vlchecklist', action='store_true')
parser.add_argument('--checkpoint-frequency', default = 6, type=int)
parser.add_argument('--vsr', action='store_true')
parser.add_argument('--lora', default = 16, type=int)
parser.add_argument('--text-lora', action='store_false')
parser.add_argument('--image-lora', action='store_false')
parser.add_argument('--prompts-lora', default = 32, type=int)
parser.add_argument('--resume', default = None, type=str)
parser.add_argument('--lr', default = 0.00005, type=float)
parser.add_argument('--prompt-attention', action='store_false')
parser.add_argument('--prompt-attention-full', action='store_false')
parser.add_argument('--lora-cross',default = 32, type=int)
parser.add_argument('--lock', action='store_true')
parser.add_argument('--epochs', default = 8, type=int)
parser.add_argument('--stop-after', default = 6, type=int)
parser.add_argument("--loss-ce", default = 1.0, type=float)

args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
blip_processor = blip_processor(image_size)

model = blip_retrieval_vg(pretrained=config['pretrained'], 
                          image_size=config['image_size'], 
                          vit=config['vit'], 
                          vit_grad_ckpt=config['vit_grad_ckpt'], 
                          vit_ckpt_layer=config['vit_ckpt_layer'], 
                          queue_size=config['queue_size'], 
                          negative_all_rank=config['negative_all_rank'], 
                          args = args)

if os.path.isfile(args.evaluate):
    checkpoint = torch.load(args.evaluate, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

model = model.to(device)
fnames = os.listdir(data_path)

for fname in fnames:
        print('=======================================================================', flush = True)
        print('=======================================', fname, '=====================', flush = True)
        json_path = os.path.join(data_path, fname)
        total = 0
        correct_img_p1 = 0
        correct_img_p2 = 0
        correct_full = 0  ###  the main task: P1 and P2 closer to Image than Negative
        # correct_text = 0

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

            image = blip_processor(image)
            image = image.unsqueeze(axis = 0).to(device)

            image_embeds = model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

            text1 = model.tokenizer(p1, padding='max_length', truncation=True, max_length=41, return_tensors="pt").to(device)
            o1 = model.text_encoder(text1.input_ids, attention_mask = text1.attention_mask, encoder_hidden_states = image_embeds, encoder_attention_mask = image_atts, return_dict = True)
            itm_o1 = model.itm_head(o1.last_hidden_state[:,0,:])
            pred_p1 =  torch.argmax(itm_o1,1).item()

            text2 = model.tokenizer(p2, padding='max_length', truncation=True, max_length=41, return_tensors="pt").to(device)
            o2= model.text_encoder(text2.input_ids, attention_mask = text2.attention_mask, encoder_hidden_states = image_embeds, encoder_attention_mask = image_atts, return_dict = True)
            itm_o2 = model.itm_head(o2.last_hidden_state[:,0,:])

            pred_p2 =  torch.argmax(itm_o2,1).item()
            text_r = model.tokenizer(neg, padding='max_length', truncation=True, max_length=41, return_tensors="pt").to(device)
            oneg= model.text_encoder(text_r.input_ids, attention_mask = text_r.attention_mask, encoder_hidden_states = image_embeds, encoder_attention_mask = image_atts, return_dict = True)
            itm_oneg = model.itm_head(oneg.last_hidden_state[:,0,:])
            
            pred_neg =  torch.argmax(itm_oneg, 1).item()
            
            total += 1
            if pred_p1 == 1 and pred_p2 == 1 and pred_neg == 0:
                correct_full += 1
            if pred_p1 == 1 and pred_neg == 0:
                correct_img_p1 += 1
            if pred_p2 == 1 and pred_neg == 0:
                correct_img_p2 += 1
            
        print(f"====== evaluation results ======", flush = True)
        ave_score = float(correct_full) / float(total)
        print(f"Accuracy image-to-text task: {ave_score}", flush = True)
        ave_score_orig_p1 = float(correct_img_p1) / float(total)
        print(f"Accuracy Image-P1-Neg: {ave_score_orig_p1}", flush = True)
        ave_score_orig_p2 = float(correct_img_p2) / float(total)
        print(f"Accuracy Image-P2-Neg: {ave_score_orig_p2}", flush = True)