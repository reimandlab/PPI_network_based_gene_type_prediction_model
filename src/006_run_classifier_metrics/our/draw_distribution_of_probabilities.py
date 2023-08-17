import numpy as np
import matplotlib.pyplot as plt

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
    probability = np.load('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_prediction_probability.npy'.format(gene_set=args.gene_subset))

    label = np.load('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_original_label.npy'.format(gene_set=args.gene_subset))

    entrez = np.load('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{gene_set}_entrez_name.npy'.format(gene_set=args.gene_subset))

    plot_list = []
    for i in range(len(label)):
        if label[i] == 0:
            plot_list.append(probability[i][1])
    plot_list.sort(reverse=True)
    print(plot_list)
    print(len(plot_list))
    x = []        
    for j in range(len(plot_list)):
        x.append(j)
        
    plt.figure(figsize=(90, 8))
    plt.plot(x,plot_list)
    plt.xlabel('Negative Genes')
    plt.ylabel('Probability of being positive samples')
    plt.savefig("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_plot/prob_neg_plot_{}.png".format(args.gene_subset))

if __name__ == "__main__":
    main()