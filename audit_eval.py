import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import argparse
import jsonlines
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import string
import random
from collections import Counter
import json

warnings.filterwarnings("ignore", category=UserWarning)

def ensemble_eval(result_store_path, distill_model_name, audit_model_paths, compare_model, lens, setting, bin_num=10, weights=None):
    probabilities = []
    labels = []

    if setting == 'kw':
        feature_file = f"{distill_model_name.replace('/', '_')}_{compare_model}_kw.json"
    else:
        feature_file = f"{distill_model_name.replace('/', '_')}_{compare_model}_kt.json"

    with jsonlines.open(os.path.join(result_store_path, feature_file)) as reader:
        for obj in reader:
            probability = obj.get('probability', '')
            label = obj.get('label', '')
            probabilities.append(probability)
            labels.append(label)

    label_counts = Counter(labels)
    lens = min(lens, label_counts[1])

    true_probs = probabilities[:label_counts[1]]
    false_probs = probabilities[label_counts[0]:]

    true_labels = labels[:label_counts[1]]
    false_labels = labels[label_counts[0]:]

    indices = random.sample(range(len(true_probs)), lens)
    probabilities = [true_probs[i] for i in indices] + [false_probs[i] for i in indices]
    labels = [true_labels[i] for i in indices] + [false_labels[i] for i in indices]

    hists = []
    bins = np.arange(0, 1.1, 1 / bin_num)
    for prob in probabilities:
        hist, _ = np.histogram(prob, bins)
        hists.append(hist)

    feature_columns = [f'feature{i + 1}' for i in range(bin_num)]
    test_data = pd.DataFrame({feature_columns[i]: [hist[i] for hist in hists] for i in range(bin_num)})

    # Collect predictions from all audit models
    all_probs = []
    for model_path in audit_model_paths:
        predictor = TabularPredictor.load(model_path)
        probs = predictor.predict_proba(test_data).iloc[:, 1].values
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    if weights:
        weights = np.array(weights)
        avg_probs = np.average(all_probs, axis=0, weights=weights)
    else:
        avg_probs = np.mean(all_probs, axis=0)

    y_pred = (avg_probs > 0.5).astype(int)
    y_true = labels

    print(f"\n[Model: {distill_model_name}] Ensemble Evaluation:")
    print(f"AUC: {roc_auc_score(y_true, avg_probs):.4f}, "
          f"ACC: {accuracy_score(y_true, y_pred):.4f}, "
          f"PRE: {precision_score(y_true, y_pred):.4f}, "
          f"REC: {recall_score(y_true, y_pred):.4f}, "
          f"f1: {f1_score(y_true, y_pred):.4f}")

    label1_probs = avg_probs[:lens]
    label0_probs = avg_probs[lens:]

    label1_is_student = (np.mean(label1_probs) > 0.5)
    label0_is_student = (np.mean(label0_probs) > 0.5)

    print(f"✅ True distilled model label=1, majority predicted as 1 → {'Distilled Model ✅' if label1_is_student else '❌ Not a distilled model'}")
    print(f"✅ Non-distilled model label=0, majority predicted as 1 → {'⚠️ Misclassified as distilled model' if label0_is_student else '✔️ Correctly recognized as non-distilled model'}")


    return [int(label1_is_student), int(label0_is_student)], [1, 0]


from model_lists import (
    shadow_model_names,
    base_model_names, cache_dir, token,
    audit_model_paths
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_store_path', type=str, default='prompts')
    parser.add_argument('--feature_store_path', type=str, default='features')
    parser.add_argument('--distill_store_path', type=str, default='distill_models')
    parser.add_argument('--teacher_model_name', type=str, default='gpt4')
    parser.add_argument('--dataset', type=str, default="legal")
    parser.add_argument('--auxiliary_dataset_name', type=str, default='legal', choices=['nq-simplified', 'HealthCareMagic', 'legal', 'finance', 'Sciq'])
    parser.add_argument('--compare_model', type=str, default="openai-community/gpt2-large")
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--bin_num', type=int, default=10)
    parser.add_argument('--method', choices=['kt', 'kw'], default='kw')
    args = parser.parse_args()
    prompt_store_path = args.prompt_store_path
    feature_store_path = args.feature_store_path
    distill_store_path = args.distill_store_path
    dataset = args.dataset
    length = args.length
    compare_model = args.compare_model


    safe_compare_model_name = compare_model.replace("/", "_")

    if dataset != args.auxiliary_dataset_name:
        safe_compare_model_name += f"_{args.auxiliary_dataset_name}"

    distill_models = []

    for base_name in base_model_names:
        base_name = base_name.split("/")[-1].replace("-", "_").lower()
        distill_name = f"distilled_{base_name}_{args.teacher_model_name}_{args.dataset}"
        distill_models.append(distill_name)

    if args.method == 'kt':

        audit_models = audit_model_paths[args.teacher_model_name][args.dataset + '_kt']
        if args.bin_num != 10:
            audit_models = audit_model_paths[args.teacher_model_name][args.dataset + '_kt_' + str(args.bin_num)]

        y_pred_all, y_true_all = [], []

        for distill_model in distill_models:
            y_pred, y_true = ensemble_eval(
                result_store_path=args.feature_store_path,
                distill_model_name=distill_model,
                audit_model_paths=audit_models,
                compare_model=safe_compare_model_name,
                lens=args.length,
                setting=args.method,
                bin_num=args.bin_num
            )
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)

        print(y_pred_all)

        print(y_true_all)

        print(f"\n\n✅ Overall Ensemble Audit Performance:")
        print(f"ACC: {accuracy_score(y_true_all, y_pred_all):.4f}, "
            f"PRE: {precision_score(y_true_all, y_pred_all):.4f}, "
            f"REC: {recall_score(y_true_all, y_pred_all):.4f}, "
            f"f1: {f1_score(y_true_all, y_pred_all):.4f}")

    if args.method == 'kw':

        audit_models = audit_model_paths[args.teacher_model_name][args.dataset + '_kw']

        if args.bin_num != 10:
            audit_models = audit_model_paths[args.teacher_model_name][args.dataset + '_kt_' + str(args.bin_num)]

        y_pred_all, y_true_all = [], []

        for distill_model in distill_models:
            y_pred, y_true = ensemble_eval(
                result_store_path=args.feature_store_path,
                distill_model_name=distill_model,
                audit_model_paths=audit_models,
                compare_model=safe_compare_model_name,
                lens=args.length,
                setting=args.method,
                bin_num=args.bin_num
            )
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_true)


        print(y_pred_all)

        print(y_true_all)

        print(f"\n\n✅ Overall Ensemble Audit Performance:")
        print(f"ACC: {accuracy_score(y_true_all, y_pred_all):.4f}, "
            f"PRE: {precision_score(y_true_all, y_pred_all):.4f}, "
            f"REC: {recall_score(y_true_all, y_pred_all):.4f}, "
            f"f1: {f1_score(y_true_all, y_pred_all):.4f}")


