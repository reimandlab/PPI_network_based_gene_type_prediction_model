import numpy as np
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gene_subset",
        default=None,
        type=str,
        required=True,
        help="hallmark gene sets.",
    )
    args = parser.parse_args()
    original_label = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{}_original_label.npy".format(args.gene_subset))

    pred_label = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{}_prediction_int_label.npy".format(args.gene_subset))
    
    entrez_names = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{}_entrez_name.npy".format(args.gene_subset))
    
    original_label_list = original_label.tolist()
    pred_label_list = pred_label.tolist()
    entrez_name_list = entrez_names.tolist()
    false_pos_list = []
    for i in range(len(original_label_list)):
        if pred_label_list[i] == 1 and original_label_list[i] != pred_label_list[i]:
            false_pos_list.append(entrez_name_list[i])
    with open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/false_positive_output/{}_false_positive_entrez_names.pkl'.format(args.gene_subset), 'wb') as f:
        pickle.dump(false_pos_list, f)

    
if __name__ == "__main__":
    main()