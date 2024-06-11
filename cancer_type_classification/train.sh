#!/bin/bash

for k in {0..9}
do

./toad-tiny.py --input_dir ./data${k} --num_classes 3 --device 'cuda:3' --num_sequences 1000 --max_length 75

done

