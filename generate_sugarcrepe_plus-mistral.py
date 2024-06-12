import os
import re
import pandas as pd
from tqdm import tqdm
import transformers
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
transformers.logging.set_verbosity_error()


## Prompts ##
persona = """ You are an instruction-following DataGenAI.
Your expertise lies in accurately interpreting and generating datasets based on given instructions. 
Your responses are expected to be precise and limited to the required output.

"""

generation_meta_prompt = """Given an input sentence describing an image caption, follow these steps:
1. Rephrase each provided sentence while focusing on preserving the original spatial relationship.
2. Pay careful attention to the positioning of objects or entities in relation to one another.
3. Ensure that the meaning remains consistent, and both the original and paraphrased sentences maintain logical coherence and grammatical correctness.

For example,
Input: Cat is under the table.
Paraphrase Idea: Rephrase the sentence to convey that the table is positioned above the cat.
Paraphrased: The table is above the cat.

Another example,
Input: The plane flies below the bright white clouds.
Paraphrase Idea: Ensure the spatial context is maintained by stating that the bright white clouds are situated above the flying plane.
Paraphrased: The plane flies below the bright white clouds.

Similarly,
Input: The third balcony is below the fourth balcony from the bottom.
Paraphrase Idea: Emphasize the consistent spatial arrangement while indicating that the fourth balcony is positioned above the third balcony from the bottom.
Paraphrased: The fourth balcony is above the third balcony from the bottom.

Remember to keep the meaning intact, and both the original and paraphrased sentences should be logically coherent and grammatically correct.

Lastly, for the final example:
Input: {caption}
Paraphrase Idea: Focus on replicating the spatial arrangement while maintaining the original meaning of the sentence,correct grammar, same meaning.
Paraphrased:"""


validation_meta_prompt = """Given a pair of captions you job is to check if the second caption is consistent with the first caption.
If it is consistent output the second caption as is, Otherwise rephrase the second caption to be consistent with the first sentence.
We are especially interested in spatial consistency and spatial relationship of the objects with each other.
examples, 
_____
caption 1 : A guy holding a skate board is speaking into a microphone.
caption 2 : The guy holding the microphone is speaking into the skateboard.
is_consistent: No, you cannot speak into a skateboard.
new_caption: The guy is speaking into the microphone while holding a skateboard.

caption 1: A family are playing frisbee on the beach.
caption 2: The frisbee is being played on the beach by a family.
is_consisitent: Yes, The caption 2 is consistent as it is the same caption written in passive voice. new_caption is same as caption 2.
new_caption: A family are playing frisbee on the beach.

caption 1: A stop sign vandalized with an "eating animals" sticker below the word "stop."
caption 2: The stop sign is below an "eating animals" sticker.
is_consistent: No, The stop cannot be below and above the sticker at the same time.
new_caption: The word "stop" sign is above an "eating animals" sticker.

caption 1:There is a phone on top of a calculator.
caption 2:A calculator lies beneath the phone.
is_consistent: Yes, the sentences are semantically equivalent. new_caption is same as caption 2.
new_caption: A calculator lies beneath the phone.
_____

Now the same for the below caption only.
caption 1: {caption}
caption 2: {caption2}
is_consistent:"""


CACHE_DIR = 'model_data'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), CACHE_DIR, "hub")


model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


## Stage 1: Generating Extra positive caption ##

def prepare_prompt(input_caption):
    prompt = generation_meta_prompt.format(caption=input_caption)
    messages = [
    {"role": "user", "content": persona},
    {"role": "assistant", "content": "Understood. Please provide me with the instructions for generating the dataset."},
    {"role": "user", "content": prompt} 
    ]
    return messages
    
def prepare_prompt_to_generate_again(context,generated_caption):
    add_to_context = [
        {"role":"assistant","content": generated_caption},
        {"role":"user","content": "The generated caption is same as the original caption. Try again!"}
        ]
    return context + add_to_context

def clean_text(text):
    matches = re.findall('.*:([\w\W]*)',text.replace('\n',''))
    if matches:
        text = matches[-1]
    if re.search('\n',text):
        text = text.split('\n')[-1].strip()
    return text.strip()

def is_duplicated(cap1, cap2):
    return cap1.strip().lower() == cap2.strip().lower()

def satisfy_minimum_overlap(cap1, cap2):
    """
    Check minimum overlap of generated caption
    incase, generation leads to unnessesary response.
    """
    overlapping_words = set(cap1.lower().split()).intersection(set(cap2.lower().split())) 
    return len(overlapping_words) > 2 

def generate_using_mistral(prompt,**config_args):
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
    prompt_length = len(inputs[0]) 
    outputs = model.generate(inputs, max_new_tokens=150, **config_args) 
    decoded_output = tokenizer.decode(outputs[0,prompt_length:], skip_special_tokens=True)
    
    return decoded_output

def generate_paraphrase_caption(caption,**config):
    input_prompt = prepare_prompt(caption)
    generated_caption = generate_using_mistral(input_prompt,**config)
    generated_caption = clean_text(generated_caption)
    return generated_caption


def generation_pipeline(caption, **config):
    """Generated captions using mistral with quality checks """
    generated_caption = generate_paraphrase_caption(caption, **config)
    success = False
    generation_attempt = 1
    while not success:
        generation_attempt += 1
        if generation_attempt > 1:
            print("Attempt:", generation_attempt, end='\r')

        if is_duplicated(caption, generated_caption):

            new_prompt = prepare_prompt_to_generate_again(prepare_prompt(caption),
                                                          generated_caption)
            generated_caption = generate_paraphrase_caption(
                new_prompt,
                do_sample=True,
                temperature=0.4)
            continue

        if not satisfy_minimum_overlap(caption, generated_caption):
            generated_caption = generate_paraphrase_caption(
                caption,
                do_sample=True,
                temperature=0.4)
            continue

        success = True
    return generated_caption

def append_generated_caption(dataframe):
    generated_captions_2 = []

    for cap in tqdm(dataframe.caption):
        out = generation_pipeline(cap)
        generated_captions_2.append(out)

    dataframe['generated_caption'] = generated_captions_2
    return dataframe


def run_stage1_on_all_files(files):
    data_dir = os.path.join(os.getcwd(), 'data', 'sugarcrepe-plus-plus-mistral')

    if not os.path.exists(data_dir):
        print('Creating Data Dir')
        os.makedirs(data_dir, exist_ok=True)
    for file in files:
        name = file.split(os.sep)[-1].split('.')[0]
        json_file = pd.read_json(file).T
        output = append_generated_caption(json_file)
        save_dir = os.path.join(data_dir,f'{name}_plus.json')
        print(f' Saving.. {save_dir}')
        output.to_json(save_dir, orient='records')



####### Stage 2: Automatic Validation #########

def prepare_prompt_for_vaidation(input_caption,generated_caption):
    """Prepares format for automatic validation"""
    prompt = validation_meta_prompt.format(caption=input_caption,caption2=generated_caption)
    messages = [
    {"role": "user", "content": prompt}]
    return messages

def check_consistency(caption,gen_caption):
    """Uses Mistral model to check consistency.
    :Returns boolean
    """
    prompt = prepare_prompt_for_vaidation(caption,gen_caption)
    validator_output = generate_using_mistral(prompt)
    is_consistent = validator_output.lower().find('yes,') == 0
    
    return is_consistent


def automatic_validation(caption, caption2):
    """Automatically validates generated caption, 
    returns a boolean and a new caption."""
    
    is_consistent = check_consistency(caption, caption2)
    is_consistent_original = is_consistent
    new_caption = ''
    if is_consistent:
        return is_consistent_original, caption2
    attempt = 0
    while not is_consistent:
        attempt += 1
        if attempt > 1:
            print("Validation attempt:", attempt, ':', new_caption) 
        new_caption = generation_pipeline(caption, do_sample=True)
        is_consistent = check_consistency(caption, new_caption)
    return is_consistent_original, new_caption

def append_checked_caption(data):
    """Checks captions and appends to Dataframe"""
    checked_captions = []
    consistent_indicator = []

    for _, (cap, generated_caption) in tqdm(data[['caption', 'generated_caption']].iterrows()):
        is_consistent, new_caption = automatic_validation(cap,generated_caption)
        consistent_indicator.append(is_consistent) 
        checked_captions.append(new_caption)
    data['consistent_indicator'] = consistent_indicator
    data['checked_caption'] = checked_captions

    return data


def run_stage2_on_all_files(files):
    """Runs automatic validation on all files and creates new validates files."""
    data_dir = os.path.join(os.getcwd(), 'data', 'sugarcrepe-plus-plus-mistral')
    if not os.path.exists(data_dir):
        print('Creating Data Dir')
        os.makedirs(data_dir, exist_ok=True)
    for file in files:
        name = file.split(os.sep)[-1].split('.')[0]
        json_file = pd.read_json(file)
        output = append_checked_caption(json_file)
        save_dir = os.path.join(data_dir,f'{name}_checked.json')
        print(f' Saving.. {save_dir}')
        output.to_json(save_dir, orient='records')
        



if __name__ == '__main__':
    
## Run Generation and Validation  ##

    parser = argparse.ArgumentParser(description='Generate SC++ using SC')
    parser.add_argument('--data_dir', help='directory where SugarCrepe is stored', required=True)
    args = parser.parse_args()
    
    original_files = glob(f'{args.data_dir}*.json')
    run_stage1_on_all_files(original_files)
    print('Running stage 2 for validating generated caption on files')
    generated_files = glob(os.path.join(os.getcwd(), 'data', 'sugarcrepe-plus-plus-mistral'))
    run_stage2_on_all_files(generated_files)
