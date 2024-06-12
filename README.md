# SUGARCREPE++
<br>
The SUGARCREPE++ dataset is created to evaluate the sensitivity of vision language models (VLMs) and unimodal language models (ULMs) to semantic and lexical alterations. The SUGARCREPE dataset consists of (only) one positive and one hard negative caption for each image. Relative to the negative caption, a single positive caption can either have low or high lexical overlap. The original SUGARCREPE only captures the high overlap case. To evaluate the sensitivity of encoded semantics to lexical alteration, we require an additional positive caption with a different lexical composition. SUGARCREPE++ fills this gap by adding an additional positive caption enabling a more thorough assessment of models’ abilities to handle semantic content
and lexical variation.


:point_right: We also host the SUGARCREPE++ dataset in huggingface dataset [here](https://huggingface.co/datasets/Aman-J/SugarCrepe_pp).

## Vision-Language model evaluation on SUGARCREPE++

We evaluate a comprehensive list of Vision-Language Models (VLMs) on SUGARCREPE++. We evaluate VLMs under two different settings: (1) image-text task (ITT) and (2) text-only task (TOT). As explained in our paper, in ITT, both the image and the corresponding triplet of captions (two positive captions and one negative caption) are provided as input. In TOT, only the text encoder of the VLMs is evaluated using the triplet of captions. Please refer to the `README_VLMs.md` in the VLMs folder for the steps to reproduce the results in the paper.

## Unimodal Language model evaluation in SUGARCREPE++

We designed SUGARCREPE++ dataset such that the overlap of semantic information between the two positive captions is always higher than between the positive and negative captions, even without considering the image. We use the text-only task (TOT) metric for ULMs as defined in the paper.

ULMs can be evaluated with the following steps:

1. Setting up the python environment for ULMs:

   `pip install -r ulm-requirements.txt`

3. This will download and evaluate models that can be run on a medium sized GPU.

   `python evaluate_ulms.py --data_dir data --models modelset1`

4. This will download and evaluate models that can be inferred on a 40GB GPU.

   `python evaluate_ulms.py --data_dir data --models modelset2 modelset3`

> Note: For Llama model, you need to request access from here [here](https://llama.meta.com/llama-downloads/); more info can be found [here](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2).

4. For help use, `python evaluate_ulms.py --help`
5. The ULM evaluation run will generate the following in the currect directory.
    * Files:
        * `ulm_results.log` contains the results of the evaluation.
        * `ulm_model_stats.log` contain model size and embedding size information of the evaluated ULMs.
    * Directory:
        * `ulm-results` contains pickle files with the sample-wise and aggregate results for each model/dataset.
        * `ulm_model_data` contains the downloaded weights from huggingface. This may be modified by changing the `CACHE_DIR` in evalute\_ulms.py
6. To summarize results of a previous run that is stored in `DIR`, run the following command.

    `python -c 'from evaluate_ulms import summarize_results; print(summarize_results("DIR"))'`

## SUGARCREPE++ Generation Pipeline

We generate an extra caption using the Mistral model and human validate the generated captions.

The steps to generate extra postive captions are shown below:

1. Download the original SUGARCREPE dataset from [here](https://github.com/RAIVNLab/sugar-crepe/tree/main/data).
2. Download the mistral-7b model [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
3. Run the following script.

    `python generate_sugarcrepe_plus-mistral.py --data_dir {directory to original sugarcrepe}`

4. The scripts will run both the stages of generation pipeline and create a new directory called `data/sugarcrepe-plus-plus-mistral` with outputs from both the stages. The 'checked\_caption' are automatic validated captions which are further considered for human validation.

> Note: Further human validation would be required to match the quality of SUGARCREPE++. The above output files would be similar in quality as files in `generated_data`.

- - -

### Contact

For further assistance email: 

*aman.jaiswal@dal.ca*

*sriharsha.d@dal.ca*

### License

Shield: [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)