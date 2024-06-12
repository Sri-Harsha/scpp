import torch
from modules.modeling import SegCLIP
import argparse
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import os
from modules.tokenization_clip import SimpleTokenizer
from seg_segmentation.config import get_config
from PIL import Image
import requests
import mmcv
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.image import tensor2imgs
import numpy as np

def Seg_model():
    description='SegCLIP on Retrieval Task'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_vis", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',)
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/images_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--max_words', type=int, default=77, help='')
    # parser.add_argument('--max_frames', type=int, default=1, help='')

    # parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.15, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    # parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="cc,coco,", type=str, help="Point the dataset to pretrain.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--lower_lr', type=float, default=0., help='lower lr for bert branch.')
    parser.add_argument('--lower_text_lr', type=float, default=0., help='lower lr for bert text branch.')

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--freeze_text_layer_num', type=int, default=0, help="Layer NO. of CLIP Text Encoder need to freeze.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/16", type=str, help="Choose a CLIP version")

    parser.add_argument('--use_vision_mae_recon', action='store_true', help="Use vision's mae to reconstruct the masked input image.")
    parser.add_argument('--use_text_mae_recon', action='store_true', help="Use text's mae to reconstruct the masked input text.")

    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight for optimizer.")
    parser.add_argument("--opt_b1", default=0.9, type=float, help="b1 for optimizer.")
    parser.add_argument("--opt_b2", default=0.98, type=float, help="b2 for optimizer.")
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--lr_start', default=0., type=float, help='initial warmup lr (Note: rate for `--lr`)')
    parser.add_argument('--lr_end', default=0., type=float, help='minimum final lr (Note: rate for `--lr`)')
    parser.add_argument('--use_pin_memory', action='store_true', help="Use pin_memory when load dataset.")
    parser.add_argument('--clip_grad', default=1., type=float, help='value of clip grad.')
    parser.add_argument('--cfg', type=str, default="seg_segmentation/default.yml", help='path to config file',)
    # parser.add_argument('--first_stage_layer', type=int, default=10, help="First stage layer.")

    parser.add_argument("--mae_vis_mask_ratio", default=0.75, type=float, help="mae vis mask ratio.")
    parser.add_argument("--mae_seq_mask_ratio", default=0.15, type=float, help="mae seq mask ratio.")

    parser.add_argument('--use_seglabel', action='store_true', help="Use Segmentation Label for Unsupervised Learning.")

    parser.add_argument('--disable_amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    # parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', type=str, help='root of output folder, the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    # parser.add_argument('--local_rank', type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument('--dataset', default='coco', choices=['voc', 'coco', 'context'], help='dataset classes')

    # parser.add_argument("--pretrained_clip_name", type=str, default="ViT-B/16", help="Name to eval", )

    parser.add_argument('--max_words', type=int, default=77, help='')
    parser.add_argument('--max_frames', type=int, default=1, help='')

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--first_stage_layer', type=int, default=10, help="First stage layer.")

    args = parser.parse_args()
    args.disable_amp = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    # if not args.do_pretrain and not args.do_train and not args.do_eval and not args.do_vis:
    #     raise ValueError("At least one of `do_pretrain`, `do_train`, `do_eval`, or `do_vis` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model_state_dict = torch.load('chkpts/segclip.bin', map_location='cpu')
    model = SegCLIP.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    return model


class Tokenize:
    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate
    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)
        if expanded_dim:
            return result[0]
        return result


class LoadImage:
    """A simple pipeline to load image."""
    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
            img = mmcv.imread(results['img'])
        elif isinstance(results['img'], np.ndarray):
            results['filename'] = None
            results['ori_filename'] = None
            img = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
            img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def build_image_pipeline(img_size=224,):
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[122.7709383, 116.7460125, 104.09373615], std=[68.5005327, 66.6321579, 70.32316305], to_rgb=True)
    test_pipeline = Compose([
        LoadImage(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, img_size),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline

def preprocess_img(img_name):
    input_img = mmcv.imread(img_name)
    data = dict(img=input_img)
    test_pipeline = build_image_pipeline(img_size=224)
    data = test_pipeline(data)
    img = data['img'][0]
    img = img[:, :224, :224]
    # img = img.view(1, 3, 224, 224)
    return img