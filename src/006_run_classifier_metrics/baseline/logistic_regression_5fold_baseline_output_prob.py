from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import csv
import os
from tqdm import tqdm


def train_vali_test_cv(X, y, args):
        print("================ Training classifier ================")

        classifier = LogisticRegressionCV(
          Cs=[0.001, 0.001 , 0.01 , 0.1 , 1 , 10 , 100 , 1000, 10000], cv=5, scoring="f1", multi_class="ovr", max_iter=300, random_state = 42)
        trained_classifier = classifier.fit(X, y)
        print("================ Predicting test set ================")
        
        label_pred = trained_classifier.predict_proba(X)
        label_pred_int = trained_classifier.predict(X)
        np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_baseline/output_probability/{gene_set}_original_label.npy'.format(gene_set=args.gene_subset),y)
        np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_baseline/output_probability/{gene_set}_prediction_probability.npy'.format(gene_set=args.gene_subset), label_pred)
        np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_baseline/output_probability/{gene_set}_prediction_int_label.npy'.format(gene_set=args.gene_subset), label_pred_int) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the input files are saved.",
    )


    parser.add_argument(
        "--gene_subset",
        default=None,
        type=str,
        required=True,
        help="hallmark gene sets.",
    )

    args = parser.parse_args()
    with open(args.input_dir + "/{set}_all_in_one_analysis_normalized.csv".format(set=args.gene_subset), "r") as f:
        rows = f.readlines()
        X = []
        y = []
        for row in rows[1:]:
            row_list = row.split(",")
            if row_list[3].strip() == "pos":
                X.append(np.array(row_list[:3], dtype=np.float64))
                y.append(1)
            if row_list[3].strip() == "neg":
                X.append(np.array(row_list[:3], dtype=np.float64))
                y.append(0)
                
    train_vali_test_cv(X, y, args)

if __name__ == "__main__":
    main()