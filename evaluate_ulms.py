import os
# Set env variables before import
CACHE_DIR = 'ulm_model_data'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(),
                                                        CACHE_DIR, 
                                                        "sentence-transformers")
from sentence_transformers import SentenceTransformer
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForCausalLM, AutoTokenizer)
from InstructorEmbedding import INSTRUCTOR
from angle_emb import AnglE
from torch import Tensor
import torch
import csv
import pandas as pd
import datetime
import pickle
from tqdm import tqdm
import argparse
from glob import glob
import logging
import numpy as np


DATA_SIZE = {
    "replace_obj": 1652,
    "swap_obj": 245,
    "replace_rel": 1406,
    "replace_att": 788,
    "swap_att": 666,
}



class ULMEncoder:
    """Encoder Class for all ULMs"""
    def __init__(self,checkpoint,instruction=None,tokenizer=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = checkpoint
        self.instruction = instruction
        self.tokenizer = tokenizer
        self.model = self.load_model()
        
    
    def load_model(self):
        if self.checkpoint == 'hkunlp/instructor-large-custom-ins': # rename to custom ins
            self.model = INSTRUCTOR('hkunlp/instructor-large').to(self.device).eval()
            self.instruction = "Represent the sentence for spatial semantics."
            
        elif self.checkpoint == 'hkunlp/instructor-large':
            self.model = INSTRUCTOR('hkunlp/instructor-large').to(self.device).eval()
            self.instruction = "Represent the sentence for cosine similarity matching."

        elif self.checkpoint == 'WhereIsAI/UAE-Large-V1':
            self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')
            self.model = self.model.cuda() if self.device == 'cuda' else self.model

        elif self.checkpoint == 'intfloat/e5-mistral-7b-instruct': # check how the weights would be downloaded
            self.model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct').to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
            self.instruction = 'Instruct: Retrieve semantically similar text.\nQuery: {}'

        elif self.checkpoint == 'SeanLee97/angle-llama-7b-nli-v2':
            self.model = AnglE.from_pretrained('NousResearch-Llama-2-7b-hf', 
                                               pretrained_lora_path='SeanLee97-angle-llama-7b-nli-v2', 
                                               pooling_strategy='last')
            self.model = self.model.cuda() if self.device == 'cuda' else self.model

        elif self.checkpoint == 'SeanLee97/angle-bert-base-uncased-nli-en-v1':
            self.model = AnglE.from_pretrained(f'SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg')
            self.model = self.model.cuda() if self.device == 'cuda' else self.model
            
        else:
            self.model = SentenceTransformer(self.checkpoint).to(self.device).eval()
        
        return self.model
            

    def encode(self,sent1,sent2,reference):
        
        if self.checkpoint in ['hkunlp/instructor-large',
                               'hkunlp/instructor-large-custom-ins']:

            embeddings = self.model.encode(
                [
                    [self.instruction, sent1],
                    [self.instruction, sent2],
                    [self.instruction, reference]
                ])

        elif self.checkpoint == 'intfloat/e5-mistral-7b-instruct':
            batch_dict = self.tokenizer([
                        self.instruction.format(sent1),
                        self.instruction.format(sent2),
                        self.instruction.format(reference)],
                        max_length=4095,
                        return_attention_mask=False,
                        padding=False,
                        truncation=True,)
            batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id]
                                            for input_ids in batch_dict['input_ids']]
            batch_dict = self.tokenizer.pad(batch_dict, 
                                            padding=True, 
                                            return_attention_mask=True, 
                                            return_tensors='pt')
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = ULMEncoder.last_token_pool(outputs.last_hidden_state,
                                            batch_dict['attention_mask']).detach().cpu().numpy()

        elif self.checkpoint == 'SeanLee97/angle-llama-7b-nli-v2':
            embeddings = self.model.encode([{'text': sent1}, 
                                            {'text': sent2},
                                            {'text': reference}])

        elif self.checkpoint == 'WhereIsAI/UAE-Large-V1':
            embeddings = self.model.encode([sent1, sent2, reference])

        elif self.checkpoint == 'SeanLee97/angle-bert-base-uncased-nli-en-v1':
            embeddings = self.model.encode([sent1, sent2, reference])
        
        else: # sentence-transformers
            embeddings = self.model.encode([sent1, sent2, reference], show_progress_bar=False)
            
        return torch.tensor(embeddings)
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        """Used by Mistral model"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
             
def sugarcrepe_plus_plus_evaluation(checkpoint_names, data_dict):
    """The samples are used without truncation using batch size of one.
    triplets: list of tripelts."""

    for checkpoint_name in checkpoint_names:
        model = ULMEncoder(checkpoint_name)
        logging_model_stats_flag = True
    
        for key in data_dict:
            triplets = data_dict[key]
            triplet_cosine_scores = []
            scores = {'TOT': []}
      
            for triplet in tqdm(triplets, desc=checkpoint_name):
                embeds = model.encode(
                    sent1=triplet['sent1'],
                    sent2=triplet['sent2'],
                    reference=triplet['reference'])
                # log model name and embedding size to a file
                if logging_model_stats_flag:
                    log_model_stats(model.model, model.checkpoint, embeds.shape[1])
                    logging_model_stats_flag = False
                
                unit_embeds = embeds / embeds.norm(dim=-1,keepdim=True)
                p1_ref = torch.dot(unit_embeds[0], unit_embeds[2]).item()
                p2_ref = torch.dot(unit_embeds[1], unit_embeds[2]).item()
                p1_p2 = torch.dot(unit_embeds[0],unit_embeds[1]).item()
                
                ulm_metric = int(p1_p2 > p2_ref and p1_p2 > p1_ref)
                scores['TOT'].append(ulm_metric)
                
                cosine_similarity_outputs = {'p1_ref':  p1_ref, 
                                             'p2_ref':  p2_ref, 
                                             'p1_p2' :  p1_p2}
                
                triplet_cosine_scores += [cosine_similarity_outputs]
                
                
            assert len(triplet_cosine_scores) == len(triplets)
            assert len(scores['TOT']) == len(triplets)
            average_scores = {k: np.mean(scores[k]) for k in scores}
            
            
            yield {
                'dataset': key,
                'checkpoint_name': checkpoint_name,
                'scores': scores,
                'raw_cosine_scores': triplet_cosine_scores,
                'average_score': average_scores}

## data loading ## 

def load_sugarcrepe_plus_plus(dir=f'data'):
    """Load sugarcrepe plus plus annotated"""
    
    column_keys = {'caption':'sent1','caption2':'sent2','negative_caption':'reference'}
    
    files = glob(f'{dir}{os.sep}*.json')
    print('All files found') if len(files) == 5 else print('All files not found')
    assert len(files) == 5

    for file in files:
        subset_name = file.split(os.sep)[-1].split('.')[0]
        data = pd.read_json(file)        
        assert len(data) == DATA_SIZE[subset_name]
        data.rename(columns=column_keys,inplace=True)
        
        yield data, file
    
## logging ##

def setup_logger(name, log_file, level=logging.INFO):
    """Setup loggers"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

model_stat_logger = setup_logger('model_stat', 'ulm_model_stats.log')

def log_model_stats(model, checkpoint_name, embedding_dim=None):
    """Log model information using the logging module."""
    size_model = 0
    total_params = 0
    if hasattr(model, 'parameters'):
        parameters = model.parameters()
    elif hasattr(model, 'base_model'):
        parameters = model.base_model.parameters()
    elif hasattr(model, 'backbone'):
        parameters = model.backbone.parameters()
    else:
        logging.error(f'{checkpoint_name}: Model not supported')
        parameters = {}

    for param in parameters:
        total_params += param.numel()
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    log_message = (
        f"Checkpoint Name: {checkpoint_name}\n"
        f"Model Class: {model.__class__.__name__}\n"
        f"Model Size: {size_model} bits\n"
        f"Model Size: {size_model / 8e6} MB\n"
        f"Model Size: {size_model / 8e9} GB\n"
        f"Total Parameters: {total_params / 1e6} Million\n"
        f"Embedding Dimension: {embedding_dim }"
    )
    model_stat_logger.info(log_message)
 
## evaluation ##

results_logger = setup_logger('ulm_results','ulm_results.log')

def run_evaluation(models, data_dict: dict, prefix=''):
    """ 
    Given checkpoints and dataset run an experiment and save results.
    """
    
    if not isinstance(models, list):
        models = [models]
        
    model_outputs = sugarcrepe_plus_plus_evaluation(
        checkpoint_names=models, data_dict=data_dict)
    
    for output in model_outputs:
        checkpoint_name = output['checkpoint_name']
        dataset_name = output['dataset']
        save_name = f'{checkpoint_name.split("/")[-1]}--{dataset_name}'
        directory = f'{prefix}-results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}{os.sep}{save_name}.pickle', 'wb') as f:
            pickle.dump(output, f)
            print(f'SAVED RESULTS for {save_name} in directory {directory}')
        
        aggregate_results = f"{dataset_name},{checkpoint_name},{output['average_score']['TOT']}"
        results_logger.info(aggregate_results)
    
    return directory

## summarize results ##

def summarize_results(dir, row_order:list=None):
    """Summarize ULM results"""
    from collections import defaultdict
    d = defaultdict(dict)
    for file in glob(f'{dir}/*'):
        obj = pickle.load(open(file,'rb'))
        dataset = obj['dataset']
        d[dataset][obj['checkpoint_name']] = obj['average_score']['TOT']
    
    if row_order is None:
        return pd.DataFrame(d).sort_index()
    output = pd.DataFrame(d).loc[row_order,:]
    output = output.loc[:,['swap_obj', 'swap_att','replace_obj', 'replace_att', 'replace_rel']]
    
    return output

## main script ##

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='select model sets')
    parser.add_argument('--data_dir', help='directory where sc++ is stored', default=f'data')
    parser.add_argument('--models', nargs='*', help='Model set to evaluate, modelset1 are small models and modelset2, modelset3 are very large models', required=True, choices=['modelset1', 'modelset2', 'modelset3'])
    parser.add_argument('--save_prefix', default='ulm', help='Directory prefix where results are saved')
    parser.add_argument('--test', default=False, action='store_true', help='Test with small subsets')
    args = parser.parse_args()

    model_set1 = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'BAAI/bge-small-en-v1.5',
    'sentence-transformers/all-MiniLM-L12-v2',
    'thenlper/gte-small',
    'SeanLee97/angle-bert-base-uncased-nli-en-v1',
    'BAAI/bge-base-en-v1.5',
    'sentence-transformers/sentence-t5-base',
    'thenlper/gte-base',
    'hkunlp/instructor-large',
    'hkunlp/instructor-large-custom-ins',
    'WhereIsAI/UAE-Large-V1',  
    'thenlper/gte-large',
    'sentence-transformers/all-roberta-large-v1',
    'sentence-transformers/stsb-roberta-large',
    'sentence-transformers/sentence-t5-xl',
    ]
    # requires GPU
    model_set2 = ['intfloat/e5-mistral-7b-instruct']
    model_set3 = ['SeanLee97/angle-llama-7b-nli-v2']
    models_to_evaluate = []
    
    if 'modelset1' in args.models:
        models_to_evaluate += model_set1
    if 'modelset2' in args.models:
        models_to_evaluate += model_set2
    if 'modelset3' in args.models:
        models_to_evaluate += model_set3
        
    assert len(models_to_evaluate) > 0, 'No models selected for evaluation'
    

    if args.test:
        
        args.save_prefix += 'test'
        sugarcrepe_plus = load_sugarcrepe_plus_plus(args.data_dir)
        sc_dict = {f'{file_dir.split(os.sep)[-1].split(".")[0]}' : data.to_dict('records')[:10] 
                    for data, file_dir in sugarcrepe_plus}
        res_dir = run_evaluation(models=models_to_evaluate, data_dict=sc_dict, prefix=args.save_prefix)
        
    else:
        
        sugarcrepe_plus = load_sugarcrepe_plus_plus(args.data_dir) 
        sc_plus_plus = {f'{file_dir.split(os.sep)[-1].split(".")[0]}' : data.to_dict('records')
                    for data, file_dir in sugarcrepe_plus}
        res_dir = run_evaluation(models=models_to_evaluate, data_dict=sc_plus_plus, prefix=args.save_prefix)
    
    results_string = summarize_results(res_dir,row_order= models_to_evaluate).to_string()
    results_logger.info('\n' + results_string)

