#!/bin/bash
source ~/environment2/bin/activate
mkdir log

python3 test_ALBEF.py > log/ALBEF.txt

python3 test_BLIP2.py > log/BLIP2.txt

python3 test_BLIP.py > log/BLIP.txt

python3 test_BLIP_SGVL.py > log/BLIP_SGVL.txt

python3 test_CLIP_SVLC.py > log/CLIP_SVLC.txt


python3 test_CyCLIP.py > log/CyCLIP.txt