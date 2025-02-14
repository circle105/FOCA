from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
import numpy as np
import pandas as pd
from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType
from merlion.utils import TimeSeries
from joblib import Parallel, delayed
import os

# Anomalies detection
def ad_predict(target, scores, mode, nu):
    if_aff = np.count_nonzero(target)
    scores = np.array(scores)
    if mode == 'direct':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_direct(target, scores, if_aff)
    elif mode == 'one-anomaly':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_one_anomaly(target, scores, if_aff)
    elif mode == 'fix':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_fix(target, scores, if_aff, nu)
    else:
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict, threshold = ad_floating(target, scores, if_aff)

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict


def ad_predict_normal(target, scores, mode, nu):
    if_aff = np.count_nonzero(target)
    scores = np.array(scores)
    if mode == 'direct':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_direct(target, scores, if_aff)
    elif mode == 'one-anomaly':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_one_anomaly(target, scores, if_aff)
    elif mode == 'fix':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_fix(target, scores, if_aff, nu)
    else:
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_floating(target, scores, if_aff)
        # affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_floating_parallel(target, scores, if_aff)

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict


# Anomalies are detected directly based on classification probabilities
def ad_direct(target, predict, if_aff):
    predict = np.int64(predict > 0.5)
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation = dict()
        affiliation["precision"] = 0
        affiliation["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation, score, score, score, predict


# For UCR dataset, there is only one anomaly period in the test set.
# Anomalies are detected based on maximum anomaly scores.
def ad_one_anomaly(target, scores, if_aff):
    scores = z_score(scores)
    threshold = np.max(scores, axis=0)
    max_number = np.sum(scores == threshold)
    predict = np.zeros(len(scores))
    # Prevents some methods from generating too many maximum values for anomaly scores.
    if max_number <= 10:
        for index, r2 in enumerate(scores):
            if r2.item() >= threshold:
                predict[index] = 1
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation_max = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation_max = dict()
        affiliation_max["precision"] = 0
        affiliation_max["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score_max = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation_max, score_max, score_max, score_max, predict


# Anomalies are detected based on anomaly scores and fixed thresholds.
def ad_fix(target, scores, if_aff, nu):
    scores = z_score(scores)
    detect_nu = 100 * (1 - nu)
    threshold = np.percentile(scores, detect_nu)
    predict = np.int64(scores > threshold)
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation_max = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation_max = dict()
        affiliation_max["precision"] = 0
        affiliation_max["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score_max = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation_max, score_max, score_max, score_max, predict

# Anomalies are detected based on anomaly scores and floating thresholds.
def ad_floating(target, scores, if_aff):
    # scores = z_score(scores)
    events_gt = convert_vector_to_events(target)
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    resolution = 10
    upper_limit = 1000
    nu_list = np.arange(1, upper_limit) / resolution
    pa_f1_list, pw_f1_list, rpa_f1_list, affiliation_f1_list = [], [], [], []
    score_list, affiliation_list = [], []
    percentile_thresholds = np.percentile(scores, 100 - nu_list)
    for i, threshold in enumerate(percentile_thresholds):
        predict = np.int64(scores > threshold)
        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            Trange = (0, len(predict))
            dic = pr_from_events(events_pred, events_gt, Trange)
            affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
            affiliation_f1_list.append(affiliation_f1)
        else:
            dic = dict()
            dic["precision"] = 0
            dic["recall"] = 0
            affiliation_f1_list.append(0)
        affiliation_list.append(dic)
        predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
        score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
        rpa_f1 = score.f1(ScoreType.RevisedPointAdjusted)
        pa_f1 = score.f1(ScoreType.PointAdjusted)
        pw_f1 = score.f1(ScoreType.Pointwise)
        rpa_f1_list.append(rpa_f1)
        pa_f1_list.append(pa_f1)
        pw_f1_list.append(pw_f1)
        score_list.append(score)
    
    affiliation_f1_list = np.nan_to_num(affiliation_f1_list)
    affiliation_max_index = np.nanargmax(affiliation_f1_list, axis=0)
    affiliation_max = affiliation_list[affiliation_max_index]
    affiliation_nu_max = nu_list[affiliation_max_index]
    print("Best affiliation quantile:", affiliation_nu_max)

    rpa_max_index = np.nanargmax(rpa_f1_list, axis=0)
    rpa_score_max = score_list[rpa_max_index]
    rpa_nu_max = nu_list[rpa_max_index]
    print('Best RPA quantile:', rpa_nu_max)

    pa_max_index = np.nanargmax(pa_f1_list, axis=0)
    pa_score_max = score_list[pa_max_index]
    pa_nu_max = nu_list[pa_max_index]
    print('Best PA quantile:', pa_nu_max)

    pw_max_index = np.nanargmax(pw_f1_list, axis=0)
    pw_score_max = score_list[pw_max_index]
    pw_nu_max = nu_list[pw_max_index]
    print('Best PW quantile:', pw_nu_max)

    threshold = np.percentile(scores, 100 - pa_nu_max)
    print('threshold:', threshold)
    predict = np.int64(scores > threshold)
    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict, threshold


# Anomaly score z_score standardization
def z_score(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    if std != 0:
        scores = (scores - mean)/std
    return scores
