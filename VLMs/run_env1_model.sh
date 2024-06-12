#!/bin/bash
source ~/environment1/bin/activate
mkdir log

python3 test_ALIGN.py > log/ALIGN.txt

python3 test_ALIP.py > log/ALIP.txt


python3 test_AltCLIP.py > log/AltCLIP.txt

python3 test_BLIP_SGVL.py > log/BLIP_SGVL.txt

python3 test_FLAVA.py > log/FLAVA.txt

python3 test_NegCLIP.py > NegCLIP.txt

python3 test_ViLT.py > log/ViLT.txt

python3 test_ViLT_ITR_COCO.py > log/ViLT_ITR_COCO.txt