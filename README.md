# FOCA
This repository provides the implementation of the FOCA: Foundation-model-based One-Class  Anomaly Detection for Time Series.The implementation uses the Merlion libraries.

## Installation

This code is based on `Python 3.10`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```

Checkpoints of pretrained weights we use in experiment : [units_x32_pretrain_checkpoint](https://github.com/mims-harvard/UniTS/releases/tag/ckpt). 


### Dataset

We acknowledge the contributors of the dataset, including  UCR and WADI. This repository already includes Merlion's data loading package ts_datasets.

UCR: [link1](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/) and [link2](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip). Download the data and unzip it in `data/ucr respectively`.

WADI: You need to apply by their [official tutorial](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) for the WADI. There are multiple versions of the WADI, we used it's newer versions:  WADI.A2_19Nov2019. Download and unzip the data in data/wadi respectively. Then run the wadi_preprocessing() functions in dataloader/data_preprocessing.py for preprocessing.


### Usage

```
# FOCA Method (dataset_name:  UCR, WADI)

python foca.py --selected_dataset <dataset_name> --device cuda --fix_seed 1

```

## Baselines

OC_SVM, IF, SR, RRCF,DAMP,LSTMED,  SVDD, COCA, AOC, TS_AD(TCC), Anomaly Transformer(AOT)

We reiterate that in addition to our method, the source code of other baselines is based on the GitHub source code provided by their papers. For reproducibility, we changed the source code of their models as little as possible. We are grateful for the work on these papers.

We consult the GitHub source code of the paper corresponding to the baseline and then reproduce it. For baselines that use the same datasets as ours, we use their own recommended hyperparameters. For different datasets, we use the same hyperparameter optimization method Grid Search as our model to find the optimal hyperparameters.

### Acknowledgements

Part of the code, especially the baseline code, is based on the following source code.

1. [Metrics:affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)
2. LSTM_ED, SR, and IF are reproduced based on [saleforce-merlion](https://github.com/salesforce/Merlion/tree/main/merlion/models/anomaly)
3. [RRCF](https://github.com/kLabUM/rrcf?tab=readme-ov-file)
4. [DAMP](https://sites.google.com/view/discord-aware-matrix-profile/documentation) and 
5. [DAMP-python](https://github.com/sihohan/DAMP)
6. [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)
7. [AOC](https://github.com/alsike22/AOC)
8. [Anomaly Transformer(AOT)](https://github.com/thuml/Anomaly-Transformer)
9. [UniTS](https://github.com/mims-harvard/UniTS)
