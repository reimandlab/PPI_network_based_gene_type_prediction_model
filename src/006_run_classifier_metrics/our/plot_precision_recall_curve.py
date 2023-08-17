from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
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
    y_pred = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{set}_prediction_probability.npy".format(set=args.gene_subset))
    y_true = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/probability_output/{set}_original_label.npy".format(set=args.gene_subset))
    
    y_pred_base = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_baseline/output_probability/{set}_prediction_probability.npy".format(set=args.gene_subset))
    y_true_base = np.load("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_baseline/output_probability/{set}_original_label.npy".format(set=args.gene_subset))


    precision, recall, threshold = precision_recall_curve(y_true, y_pred[:, 1])
    auc_score = auc(recall, precision)
    
    precision_base, recall_base, threshold_base = precision_recall_curve(y_true_base, y_pred_base[:, 1])
    auc_score_base = auc(recall_base, precision_base)


    plt.plot(recall, precision, 'b', label='Our model'.format(set=args.gene_subset))
    plt.plot(recall_base, precision_base, 'g', label='baseline'.format(set=args.gene_subset))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precesion-Recall curve {set}".format(set=args.gene_subset))

    plt.text(0.6, 0.8, "AUC: {}".format(str(round(auc_score, 4))))
    plt.text(0.6, 0.9, "AUC_base: {}".format(str(round(auc_score_base, 4))))
    
    plt.savefig("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/run_classifier_for_metrics/logistic_regression_n2v/PR_curve/PR_curve_{}.jpg".format(args.gene_subset), dpi=300)
    
if __name__ == "__main__":
    main()