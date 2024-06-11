# DECIDIA: Deep-learning-based approach for Early Cancer Interception and DIAgnosis.

## Introduction
Early cancer diagnosis from bisulfite-treated cell-free DNA (cfDNA) fragments require tedious data analytical procedures. Here, we present a Deep-learning-based approach for Early Cancer Interception and DIAgnosis (DECIDIA) that can achieve accurate cancer diagnosis exclusively from bisulfite-treated cfDNA sequencing fragments. DECIDIA relies on feature representation learning of DNA fragments and weakly supervised learning for classification.   

## System requirements
- Operating systems: CentOS 7.
- [Python](https://docs.conda.io/en/latest/miniconda.html) (version == 3.7).
- [PyTorch](https://pytorch.org) (version == 1.13.1+cu116).
- [transformers](https://huggingface.co/docs/transformers/index) (version == 4.28.1).

This example was tested with the following environment. However, it should work on the other platforms. 

## Installation guide
- Following instruction from [miniconda](https://docs.conda.io/en/latest/miniconda.html) to install Python.
- Use the following command to install required packages.
```bash
# Install with GPU support. Check https://pytorch.org for more information. 
#+The following cmd install PyTorch compiled with cuda 118. 
pip install torch --index-url https://download.pytorch.org/whl/cu118

# If GPU not available, install the PyTorch compiled for CPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers, tokenizers and prettytable
pip install transformers==4.28.1 tokenizers==0.13.3 prettytable
```

- The installation process will take about an hour. This heavily depends on your network bandwidth.

## Demo
- Clone `DECIDIA` locally from Github.
```bash
git clone https://github.com/deeplearningplus/DECIDIA.git
```
- Instruction to train cancer detection model:
```bash
bash cancer_detection.sh
```

- Instruction to train cancer type classification model
```python
bash cancer_type_classification.sh
```

The outputs include log file `log*.txt`, checkpoint file `*pt` and prediction probabilities `*csv` on the `input_dir`.

## How to run on your own data
prepare the raw sequence data in the same format as `./data_efficiency/data0/trn.csv.gz` and run `cancer_detection.sh` or `cancer_type_classification.sh`.
