#!/bin/bash

for k in {0..9}
do

cd data${k} && cp ../split_data.py ./ && python split_data.py && cd ../ &

done
