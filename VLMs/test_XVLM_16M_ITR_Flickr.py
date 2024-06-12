import sys, torch, os, json
sys.path.append(os.path.join(os.getcwd(), 'XVLM'))  ######  provide path to the XVLM folder
sys.path.append(os.path.join(os.getcwd(), 'XVLM/configs'))
from models.model_retrieval import XVLM 
from models.tokenization_bert import BertTokenizer
from torchvision import transforms
from ruamel.yaml import YAML  #import ruamel.yaml as yaml
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

current_dir = os.getcwd()

img_path = os.path.join(current_dir, 'images/') #path to the images folder
data_path =  os.path.join(current_dir, 'data/') #path to folder with caption files
config_path = Path('XVLM/configs/Pretrain_XVLM_base_4m.yaml')
chkpt_path = os.path.join(current_dir, 'chkpts/xvlm_16m_itr_flickr.pth')

yaml = YAML(typ='safe')
config = yaml.load(config_path)
os.chdir('XVLM')
model = XVLM(config=config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_pretrained(chkpt_path, config, is_eval=False)
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], 
        config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize])

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


            image = test_transform(image).unsqueeze(axis = 0)
            image = image.to(device)
            image_feat = model.vision_encoder(image)
            img_feats = model.vision_proj(image_feat[:, 0, :])
            img_feats = F.normalize(img_feats, dim=-1)

            p1 = p1.rstrip('\n').strip(' ').lower()
            p1_words = p1.split(' ')
            if len(p1_words) > 30:
                p1 = ' '.join(p1_words[:30])
            p1_in = tokenizer(p1, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
            p1_out = model.text_encoder(p1_in.input_ids, attention_mask=p1_in.attention_mask, mode='text')
            p1_feats = p1_out.last_hidden_state
            p1_feats = F.normalize(model.text_proj(p1_feats[:, 0, :]))

            p2 = p2.rstrip('\n').strip(' ').lower()
            p2_words = p2.split(' ')
            if len(p2_words) > 30:
                p2 = ' '.join(p2_words[:30])
            p2_in = tokenizer(p2, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
            p2_out = model.text_encoder(p2_in.input_ids, attention_mask=p2_in.attention_mask, mode='text')
            p2_feats = p2_out.last_hidden_state
            p2_feats = F.normalize(model.text_proj(p2_feats[:, 0, :]))

            neg = neg.rstrip('\n').strip(' ').lower()
            neg_words = neg.split(' ')
            if len(neg_words) > 30:
                neg = ' '.join(neg_words[:30])
            neg_in = tokenizer(neg, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
            neg_out = model.text_encoder(neg_in.input_ids, attention_mask=neg_in.attention_mask, mode='text')
            neg_feats = neg_out.last_hidden_state
            neg_feats = F.normalize(model.text_proj(neg_feats[:, 0, :]))

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
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