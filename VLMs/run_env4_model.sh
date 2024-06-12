#!/bin/bash
source ~/environment4/bin/activate
mkdir log

python3 test_XVLM_4M.py > log/XVLM_4M.txt

python3 test_XVLM_16M.py > log/XVLM_16M.txt

python3 test_XVLM_16M_ITR_COCO.py > log/XVLM_16M_ITR_COCO.txt

python3 test_XVLM_16M_ITR_Flickr.py > log/XVLM_16M_ITR_Flickr.txt