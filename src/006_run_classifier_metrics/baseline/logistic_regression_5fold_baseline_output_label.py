from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import csv
import os
from tqdm import tqdm

def train_vali_test_cv(X, y, args):
    for i in tqdm(range(5)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        print("================ Training classifier ================")

        classifier = LogisticRegressionCV(
          Cs=[0.001, 0.001 , 0.01 , 0.1 , 1 , 10 , 100 , 1000, 10000], cv=5, scoring="f1", multi_class="ovr", max_iter=300, random_state = 42)
        trained_classifier = classifier.fit(X_train, y_train)
        print("================ Predicting test set ================")
        y_pred = trained_classifier.predict(X_test)

        roc_auc_score_inside = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
        recall = recall_score(y_test, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
        precision = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary')

        print("feature_importance: ", trained_classifier.coef_)
        
        if os.path.exists(args.output_dir + "/{set}_performance_metrics.csv".format(set=args.gene_subset)):
            with open(args.output_dir + "/{set}_performance_metrics.csv".format(set=args.gene_subset), 'a') as f2:
                writer = csv.writer(f2)
                writer.writerow([roc_auc_score_inside, f1, recall, precision])
        else:
            with open(args.output_dir + "/{set}_performance_metrics.csv".format(set=args.gene_subset), 'w') as f2:
                writer = csv.writer(f2)
                header = ["roc_auc", "f1", "recall","precision"]
                writer.writerow(header)
                writer.writerow([roc_auc_score_inside, f1, recall, precision])

    

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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the output files are saved.",
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