from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import csv
import os
from tqdm import tqdm
import pickle

def train_vali_test_cv(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    print("================ Training classifier ================")

    classifier = LogisticRegressionCV(
      Cs=[0.001 , 0.01 , 0.1 , 1 , 10 , 100 , 1000], cv=5, scoring="f1", multi_class="ovr", max_iter=3000)
    trained_classifier = classifier.fit(X_train, y_train)
    
    print("================ Predicting test set ================")
    
    y_pred = trained_classifier.predict(X_test)

    roc_auc_score_inside = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_test, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
    precision = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary')

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
    parser.add_argument(
        "--num",
        default=None,
        type=str,
        required=True,
        help="Path where the output files are saved.",
    )
    parser.add_argument(
        "--cancer_gene_data_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the cancer data are saved.",
    )
    args = parser.parse_args()
    match_dict_int_entrez = pickle.load(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all_interaction_int_entrez.pkl', 'rb'))
    
    with open(args.cancer_gene_data_dir) as f1:
        lines = f1.readlines()
        msigDB_dict = {}
        for line in lines:
            msigDB_dict[line.split('\t')[0]] = []
            pos_genes = line.split('\t')[2:]
            for each_gene in pos_genes:
                msigDB_dict[line.split('\t')[0]].append(each_gene.strip()) #hallmark_name: entrez
    
    if args.gene_subset == "all":
        with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/node_embeddings_original/{num}/node_embeddgins_origin_512_40_4.emb".format(num = args.num), "r") as f:
                  samples_vector = []
                  samples_label = []

                  rows = f.readlines()
                  for row in rows[1:]:
                      vector = row.strip().split(" ")[1:]
                      node_name = row.strip().split(" ")[0]
                      if match_dict_int_entrez[node_name] in sum(msigDB_dict.values(), []):
                          samples_vector.append(vector)
                          samples_label.append(1)
                      else:
                          samples_vector.append(vector)
                          samples_label.append(0)

    else:
        with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/node_embeddings_original/{num}/node_embeddgins_origin_512_40_4.emb".format(num=args.num), "r") as f:
                  samples_vector = []
                  samples_label = []
                  rows = f.readlines()
                  for row in rows[1:]:
                      vector = row.strip().split(" ")[1:]
                      node_name = row.strip().split(" ")[0]
                      if match_dict_int_entrez[node_name] in msigDB_dict[args.gene_subset]:
                          samples_vector.append(vector)
                          samples_label.append(1)
                      else:
                          samples_vector.append(vector)
                          samples_label.append(0)
                        
    train_vali_test_cv(np.array(samples_vector, dtype=np.float64), samples_label, args)

if __name__ == "__main__":
    main()