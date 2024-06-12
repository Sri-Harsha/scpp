import torch
from src.open_alip import create_model, image_transform
def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed

def get_transform(image_size):
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    
    preprocess = image_transform(image_size, is_train=False, mean=image_mean, std=image_std)
    return preprocess