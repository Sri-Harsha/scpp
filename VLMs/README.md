## Vision-Language Model Evaluation on SUGARCREPE++

To evaluate the comprehensive set of VLMs on SUGARCREPE++, follow the below steps:
1. First download the MSCOCO images:
    > `python3 download_coco_images.sh`

    The above code will creta a folder "images" and downloads all the images into this folder.


2. To download the checkpoints of different VLMs:
    > `pip install gdown`
    `python3 download_checkpoints.sh`

    The above command will create a folder called "chkpts" and will download model checkpoints which require about 25 GB free space.


3. As the requirements for each VLM are different, We recommend installing four different environments (preferably virtual). The details of each environment, and the models that can be evaluated in a specific environment are provided below.

    - Environment-1: Create an environment using requirements_1.txt file. This environment is used to obtain results for the following set of VLMs: All the different variants of CLIP, ALIGN, ALIP, AltCLIP, BLIP-SGVL, FLAVA, NegCLIP, ViLT and ViLT-ITR-COCO.

        Setting up the python environment-1:
        > `pip install -r requirements_1.txt`

        To evaluate the following VLMs -- ALIGN, ALIP, AltCLIP, BLIP-SGVL, FLAVA, NegCLIP, ViLT and ViLT-ITR-COCO:
        > `bash run_env1_models.sh`

        Code to generate the results reported in Table 5 (Main paper) and Table 10 (Appendix).
        > `bash run_variants_clip.sh`


    - Environment-2: Create an environment using requirements_2.txt file. This environment is used to obtain results for the following set of VLMs: ALBEF, BLIP, BLIP-2, CLIP-SVLC and CyCLIP.

        Setting up the python environment-2:
        > `pip install -r requirements_2.txt`

        To evaluate the following VLMs -- ALBEF, BLIP, BLIP-2, CLIP-SVLC and CyCLIP:
        > `bash run_env2_models.sh`


    - Environment-3: Create an environment using requirements_3.txt file. This environment is used to obtain results for the following set of VLMs: SegCLIP

        Setting up the python environment-3:
        > `pip install -r requirements_3.txt`

        To evaluate the following VLMs -- SegCLIP:
        > `bash run_env3_models.sh`


    - Environment-4: Create an environment using requirements_4.txt file. This environment is used to obtain results for the following set of VLMs: XVLM-4M, XVLM-16M, XVLM-16M-ITR-COCO and XVLM-16M-ITR-Flickr.

        Setting up the python environment-4:
        > `pip install -r requirements_4.txt`

        To evaluate the following VLMs -- XVLM-4M, XVLM-16M, XVLM-16M-ITR-COCO and XVLM-16M-ITR-Flickr:
        > `bash run_env4_models.sh`
    
4. Understanding the Outputs:
    - Accuracy image-to-text task: Accuracy reported in the paper for image-to-text task where both caption1 and caption2 are closer to image compared to the negative caption.
    - Accuracy text-only task: Accuracy of the text-only task, where the accuracy is computed based on the number of samples where Caption1 is closer to caption2 compared to the negative caption.
    - Accuracy Image-P1-Neg: Accuracy of the number of samples where caption1 is closer to image than the negative caption; Caption2 not considered.
    - Accuracy Image-P2-Neg: Accuracy of the number of samples where caption2 is closer to image than the negative caption; Caption1 not considered
