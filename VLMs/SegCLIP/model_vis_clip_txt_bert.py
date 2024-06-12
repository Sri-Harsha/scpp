import torch  #import os, glob #import numpy as np
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
import clip

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Bert_Clip_model(nn.Module):
	#######  Bert Model for Classification Tasks  ##
	def __init__(self, mod1_name, mod2_name):  #sent1_key, sent2_key,
		super(Bert_Clip_model, self).__init__()
		bert_txt_dim = 384  #768 if "base" in mod1_name else 1024
		clip_img_dim = 512

		self.bert_sent = AutoModel.from_pretrained(mod1_name) ##'sentence-transformers/all-MiniLM-L6-v2')
		# self.tokenizer_bert = AutoTokenizer.from_pretrained(mod1_name) #'sentence-transformers/all-MiniLM-L6-v2')
		self.clip_model, _ = clip.load(mod2_name, jit=False)
		# self.processor_clip = CLIPProcessor.from_pretrained(mod_name)   ### "openai/clip-vit-base-patch32"

		# self.sent1_key = sent1_key
		# self.sent2_key = sent2_key
		self.out_proj_layer = nn.Linear(clip_img_dim, bert_txt_dim)

	def forward(self, inp_imgs, inp_ids, tok_ids, att_mask):
		# clip_inputs = self.processor_clip(text=captions, images=images, return_tensors="pt", padding=True)
		logits_per_images = self.clip_model.encode_image(inp_imgs)
		# print('shape of image logits: ', logits_per_images.shape, flush = True)
		with torch.no_grad():
			bert_outputs = self.bert_sent(input_ids = inp_ids, token_type_ids = tok_ids, attention_mask = att_mask)
		text_out = mean_pooling(bert_outputs, att_mask)
		text_out = F.normalize(text_out, p=2, dim=1)
		# lhs_clip = o_clip.last_hidden_state
		
		img_out = self.out_proj_layer(logits_per_images)

		return img_out, text_out