mkdir chkpts
cd chkpts

###################################################################
######  for this code to run-- install gdown as: pip install gdown
###################################################################

## download XVLM_4M model
gdown --fuzzy "https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing"
mv 4m_base_model_state_step_199999.th xvlm_4m.th

## download XVLM_16M model
gdown --fuzzy "https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view?usp=sharing"
mv 16m_base_model_state_step_199999.th xvlm_16m.th

## download XVLM_16M_ITR_flicker
gdown --fuzzy "https://drive.google.com/file/d/1vhdtH3iFaoZuMqOGm-8YM-diPWVfRJzv/view?usp=share_link" 
mv checkpoint_best.pth xvlm_16m_itr_flickr.pth

## download XVLM_16M_ITR_MSCOCO
gdown --fuzzy "https://drive.google.com/file/d/1bv6_pZOsXW53EhlwU0ZgSk03uzFI61pN/view?usp=share_link"
mv checkpoint_9.pth xvlm_16m_itr_coco.pth

## Download ALIP model
gdown --fuzzy "https://drive.google.com/file/d/1AqSHisCKZOZ16Q3sYguK6zIZIuwwEriE/view?usp=sharing"  

## Download NegCLIP model
gdown 1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ

## Download SegCLIP
wget https://github.com/ArrowLuo/SegCLIP/releases/download/checkpoint_v0/segclip.bin

## CLIP-SVLC
gdown --fuzzy "https://drive.google.com/file/d/1k-JAVRnyX0UGSY0Ng5EA1vD4GrhbiVZ2/view?usp=share_link"

## BLIP-SGVL
gdown --fuzzy "https://drive.google.com/file/d/13jzpcLgGalO3hkiqVwziNAlCEZD90ENN/view?usp=sharing"

##  CyCLIP
gdown --fuzzy https://drive.google.com/file/d/1nF33F3yjtiWr3bgllBXk5Wf07Uo7Uv9G/view?usp=share_link
mv best.pt cyclip_3M.pt