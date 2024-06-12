#!/bin/bash
source ~/environment1/bin/activate
mkdir log
python3 test_variants_clip.py > log/variants_clip.txt