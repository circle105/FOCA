import os
import torch
import random
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType

from dataloader import *
from models.FOCA.trainer.trainer import Trainer
from models.FOCA.network.model import base_Model
from models.reasonable_metric import reasonable_accumulator
from utils import print_object, set_seed
from ts_datasets.ts_datasets.anomaly import IOpsCompetition, UCR

start_time = datetime.now()
home_dir = os.getcwd()

parser = argparse.ArgumentParser(description="SVDD unsupervised training")
parser.add_argument("--fix_seed", default=1, type=int, help="seed value")
parser.add_argument("--selected_dataset", default="UCR", type=str, help="Dataset of choice: IOpsCompetition, UCR, SWaT, WADI")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
args = parser.parse_args()

device = torch.device(args.device)
data_type = args.selected_dataset
method = "foca"
selected_dataset = args.selected_dataset

exec(f"from conf.{method}.{data_type}_Configs import Config as Configs")
configs = Configs()

SEED = args.fix_seed

print(f"Dataset: {data_type}")
print(f"Method:  {method}")
print(f"Random Seed:  {SEED}")

# Load datasets
if selected_dataset == "WADI":
    model_num = 1
else:
    if selected_dataset == "UCR":
        dt = UCR()
    model_num = len(dt)

all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
all_anomaly_num, all_test_scores_reasonable = [], []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
detect_list = np.zeros(model_num)

for idx in tqdm(range(model_num)):
    set_seed(SEED)
    configs.fix_seed = SEED
    if selected_dataset == "WADI":
        train_data, test_data, train_labels, test_labels = wadi()
    else:
        time_series, meta_data = dt[idx]
        train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)

    print(">" * 32, len(train_data))
    print(">" * 32, len(test_data))

    train_dl, val_dl, test_dl,_, test_anomaly_window_num = data_generator1(train_data, test_data, train_labels, test_labels, SEED, configs)
    print(f"samples\t|train {len(train_dl.dataset)}\t|val {len(val_dl.dataset)}\t|test {len(test_dl.dataset)}")

    model = base_Model(configs).to(device)

    trainer = Trainer(model, device, configs, idx, SEED)
    test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = trainer.train(train_dl, val_dl, test_dl)

    all_anomaly_num.append(test_anomaly_window_num)
    all_test_scores_reasonable.append(score_reasonable)
    all_test_aff_precision.append(test_aff["precision"])
    all_test_aff_recall.append(test_aff["recall"])
    all_test_aff_score.append(test_aff)
    all_test_rpa_score.append(test_rpa_score)
    all_test_pa_score.append(test_pa_score)
    all_test_pw_score.append(test_pw_score)

all_anomaly_num = np.array(all_anomaly_num)
sum_anomaly_num = np.sum(all_anomaly_num)
all_test_aff_precision = np.array(all_test_aff_precision)
all_test_aff_precision = all_test_aff_precision * all_anomaly_num / sum_anomaly_num
test_aff_precision = np.nansum(all_test_aff_precision)
all_test_aff_recall = np.array(all_test_aff_recall)
all_test_aff_recall = all_test_aff_recall * all_anomaly_num / sum_anomaly_num
test_aff_recall = np.nansum(all_test_aff_recall)
if test_aff_precision + test_aff_recall == 0:
    test_aff_f1 = 0
else:
    test_aff_f1 = 2 * (test_aff_precision * test_aff_recall) / (test_aff_precision + test_aff_recall)

total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
ucr_accuracy = total_test_scores_reasonable.get_all_metrics()

print(">" * 32)
if configs.dataset == "UCR":
    print("UCR metrics:\n", f"accuracy: {ucr_accuracy}\n")
print(
    "affiliation metrics:\n",
    f"Precision: {test_aff_precision:.5f}\n",
    f"Recall:    {test_aff_recall:.5f}\n",
    f"F1 Scores: {test_aff_f1:.5f}\n",
    "Revised-point-adjusted metrics:\n",
    f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
    f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n",
    f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
    "Point-adjusted metrics:\n",
    f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
    f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
    f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
    "Point-wise metrics:\n",
    f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
    f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n",
    f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n" "NAB Scores:\n",
    f"NAB Score (balanced):       {total_test_pa_score.nab_score():.5f}\n",
    f"NAB Score (high precision): {total_test_pa_score.nab_score(fp_weight=0.22):.5f}\n",
    f"NAB Score (high recall):    {total_test_pa_score.nab_score(fn_weight=2.0):.5f}\n",
    f"seed: {SEED}\n",
    "config setup:\n",
)

str_conf = print_object(configs,except_list=["fix_seed"])
train_time = datetime.now() - start_time
print(f"Training time is : {train_time}")

path = "./results/test"
if not os.path.exists(path):
    os.makedirs(path)

cur_time = datetime.now().strftime("%Y-%m-%d")
summary = os.path.join(path, f"{method}_{selected_dataset}_summary_{cur_time}.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = SEED
df.loc["Train Time", model_name] = train_time
df.loc["UCR Accuracy", model_name] = round(ucr_accuracy["accuracy"], 5)
df.loc["Affiliation Precision", model_name] = round(test_aff_precision, 5)
df.loc["Affiliation Recall", model_name] = round(test_aff_recall, 5)
df.loc["Affiliation F1", model_name] = round(test_aff_f1, 5)
df.loc["RPA Precision", model_name] = round(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA Recall", model_name] = round(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA F1", model_name] = round(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted), 5)
df.loc["PA Precision", model_name] = round(total_test_pa_score.precision(ScoreType.PointAdjusted), 5)
df.loc["PA Recall", model_name] = round(total_test_pa_score.recall(ScoreType.PointAdjusted), 5)
df.loc["PA F1", model_name] = round(total_test_pa_score.f1(ScoreType.PointAdjusted), 5)
df.loc["Point-wise Precision", model_name] = round(total_test_pw_score.precision(ScoreType.Pointwise), 5)
df.loc["Point-wise Recall", model_name] = round(total_test_pw_score.recall(ScoreType.Pointwise), 5)
df.loc["Point-wise F1", model_name] = round(total_test_pw_score.f1(ScoreType.Pointwise), 5)
df.loc["NAB Score (balanced)", model_name] = round(total_test_rpa_score.nab_score(), 5)
df.loc["NAB Score (high precision)", model_name] = round(total_test_rpa_score.nab_score(fp_weight=0.22), 5)
df.loc["NAB Score (high recall)", model_name] = round(total_test_rpa_score.nab_score(fn_weight=2.0), 5)


df.to_csv(summary, index=True)

