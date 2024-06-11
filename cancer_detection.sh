#!/bin/bash

for k in {0..9}
do

for num_sequences in 200 400 600 800 1000
do

for num_train_patients in 200 400 600 800 1000
do

python ./cancer_detection.py --input_dir cancer_detection/data${k} --num_classes 2 --num_sequences $num_sequences --device 'cuda:0' --num_train_patients $num_train_patients

done

done

done
