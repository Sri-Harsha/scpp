
# Vision-Language Model Evaluation on SUGARCREPE++

To evaluate a comprehensive set of VLMs on SUGARCREPE++, follow the steps below:

1. **Download MSCOCO images:**

   ```bash
   python3 download_coco_images.sh
   ```
   
   This command creates a folder "images" and downloads all the images into it.

2. **Download VLM checkpoints:**

   ```bash
   pip install gdown
   python3 download_checkpoints.sh
   ```

   This command creates a folder called "chkpts" and downloads model checkpoints, requiring approximately 25 GB of free space.

3. **Set up Python environments:**
    - **Environment-1:** Install requirements from `requirements_1.txt`. This environment evaluates CLIP (and its variants), ALIGN, ALIP, AltCLIP, BLIP-SGVL, FLAVA, NegCLIP, ViLT, and ViLT-ITR-COCO.
       ```bash
       pip install -r requirements_1.txt
       bash run_env1_model.sh
       ```
       Use `bash run_variants_clip.sh` to generate results reported in the paper.
       
    - **Environment-2:** Install requirements from `requirements_2.txt`. This environment evaluates ALBEF, BLIP, BLIP-2, CLIP-SVLC, and CyCLIP.
       ```bash
       pip install -r requirements_2.txt
       bash run_env2_model.sh
       ```
    
    - **Environment-3:** Install requirements from `requirements_3.txt`. This environment evaluates SegCLIP.
       ```bash
       pip install -r requirements_3.txt
       bash run_env3_model.sh
       ```
    
    - **Environment-4:** Install requirements from `requirements_4.txt`. This environment evaluates XVLM-4M, XVLM-16M, XVLM-16M-ITR-COCO, and XVLM-16M-ITR-Flickr.
       ```bash
       pip install -r requirements_4.txt
       bash run_env4_model.sh
       ```

4. **Understanding the Outputs:**
   - **Accuracy image-to-text task:** Accuracy reported in the paper for image-to-text task where both caption1 and caption2 are closer to image compared to the negative caption.
   - **Accuracy text-only task:** Measures accuracy based on the number of samples where Caption1 is closer to caption2 compared to the negative caption.
   - **Accuracy Image-P1-Neg:** Measures accuracy where caption1 is closer to the image than the negative caption; Caption2 not considered.
   - **Accuracy Image-P2-Neg:** Measures accuracy where caption2 is closer to the image than the negative caption; Caption1 not considered.

___


