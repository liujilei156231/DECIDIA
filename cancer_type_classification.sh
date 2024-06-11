#!/bin/bash

for k in {0..9}
do

python ./cancer_type_classification.py --input_dir cancer_type_classification/data${k} --num_classes 3 --device 'cuda:0' --num_sequences 1000 --max_length 75

done

