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
    gene_entrez = []
    label = []
    for i in range(len(y)):
        gene_entrez.append(y[i][1])
        label.append(y[i][0])

    print("================ Training classifier ================")

    classifier = LogisticRegressionCV(
      Cs=[0.001 , 0.01 , 0.1 , 1 , 10 , 100 , 1000], cv=5, scoring="f1", multi_class="ovr", max_iter=3000)
    trained_classifier = classifier.fit(X, label)
    
    print("================ Predicting test set ================")
    
    label_pred = trained_classifier.predict_proba(X)
    label_pred_int = trained_classifier.predict(X)
    print(trained_classifier.classes_)
    np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_original_label.npy'.format(gene_set=args.gene_subset),label)
    np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_prediction_probability.npy'.format(gene_set=args.gene_subset), label_pred)
    np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_prediction_int_label.npy'.format(gene_set=args.gene_subset), label_pred_int)  
    np.save('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_entrez_name.npy'.format(gene_set=args.gene_subset), gene_entrez)


def main():
    parser = argparse.ArgumentParser()
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
                          samples_label.append([1,match_dict_int_entrez[node_name]])
                      else:
                          samples_vector.append(vector)
                          samples_label.append([0,match_dict_int_entrez[node_name]])

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
                          samples_label.append([1,match_dict_int_entrez[node_name]])
                      else:
                          samples_vector.append(vector)
                          samples_label.append([0,match_dict_int_entrez[node_name]])
                
    train_vali_test_cv(np.array(samples_vector, dtype=np.float64), samples_label, args)

if __name__ == "__main__":
    main()