import gensim
import numpy as np
from numpy.linalg import norm
from gensim.models import Word2Vec
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances
import os
import csv



def load__match_dict(n2v_match_dict_dir):
    with open(n2v_match_dict_dir, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def mean_matrix_off_diagnol(A):

    matrix_without_diagnol = A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
    mean = np.mean(matrix_without_diagnol)
    return mean


def plot_embedding_similarity(node_embeddings_2d, labels, save_path, args, node_names_entrez, node_names_ensebl):
    node_emb_2d_pos = []
    node_names_pos_entrez = []
    node_emb_2d_neg = []
    node_names_neg_entrez = []
    for i in range(len(labels)):
        if labels[i] == 1:
            node_emb_2d_pos.append(node_embeddings_2d[i, :])
            
        else:
            node_emb_2d_neg.append(node_embeddings_2d[i, :])
            node_names_neg_entrez.append(node_names_entrez[i])

    pos_neg_similarity_matrix = cosine_similarity(node_emb_2d_pos, node_emb_2d_neg)
    print("neg_pos cos similarity: ", pos_neg_similarity_matrix.shape, len(pos_neg_similarity_matrix))

    pos_neg_euc_dis_matrix = euclidean_distances(node_emb_2d_pos, node_emb_2d_neg)
    print("pos_neg euc distance: ", pos_neg_euc_dis_matrix.shape, len(pos_neg_euc_dis_matrix))


    with open(save_path + "/{set}_biological_analysis_{dim}_{walk_len}_{num}_eu_dis_distribution_0.csv".format(set=args.gene_subset,dim=args.dim, walk_len=args.walk_len, num=args.num_walks), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['eu_dis', "sample_index"])
        count = 0
        for i in range(len(pos_neg_euc_dis_matrix)):
            row = [pos_neg_euc_dis_matrix[i][0], count]
            writer.writerow(row)
            count += 1
        
    
    
    
    average_euclidean_dis_of_all_pos_to_a_negative_list = np.median(pos_neg_euc_dis_matrix, axis=0).tolist() 
    average_cos_sim_of_all_pos_to_a_negative_list = np.median(pos_neg_similarity_matrix, axis=0).tolist() 
    
    
    sorted_eu_dis_list, eu_correspond_node_names_entrez = (list(t) for t in zip(*sorted(zip(average_euclidean_dis_of_all_pos_to_a_negative_list, node_names_neg_entrez))))
    
    sorted_cos_sim_list, cos_correspond_node_names_entrez = (list(t) for t in zip(*sorted(zip(average_cos_sim_of_all_pos_to_a_negative_list, node_names_neg_entrez))))  
        
    header = ['Hallmark gene set', 'eu_dis', "id"]

    with open(save_path + "/{set}_biological_analysis_{dim}_{walk_len}_{num}_eu_dis.csv".format(set=args.gene_subset, dim=args.dim, walk_len=args.walk_len, num=args.num_walks), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(sorted_eu_dis_list)):
            row = [args.gene_subset, sorted_eu_dis_list[i], eu_correspond_node_names_entrez[i]]
            writer.writerow(row)
        
    with open(save_path + "/{set}_biological_analysis_{dim}_{walk_len}_{num}_cos_sim.csv".format(set=args.gene_subset, dim=args.dim, walk_len=args.walk_len, num=args.num_walks), 'w') as f5:
        writer = csv.writer(f5)
        writer.writerow(header)
        for j in range(len(sorted_cos_sim_list)):
            row = [args.gene_subset, sorted_cos_sim_list[j], cos_correspond_node_names_entrez[j]]
            writer.writerow(row)
            

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--cancer_gene_data_dir",
        default=None,
        type=str,
        required=True,
        help="Path where the cancer data are saved.",
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
        help=".",
    )
    
    parser.add_argument(
        "--dim",
        default=None,
        type=str,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--walk_len",
        default=None,
        type=str,
        required=True,
        help=".",
    )
    parser.add_argument(
        "--num_walks",
        default=None,
        type=str,
        required=True,
        help=".",
    )

    args = parser.parse_args()
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all_one_to_one.pkl", 'rb') as f0:
        match_dict_ensembl_entrez_dict = pickle.load(f0)
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all.pkl", 'rb') as f:
        match_dict = pickle.load(f)

    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all_int_ensembl.pkl", 'rb') as f2:
        match_dict_int_ensembl = pickle.load(f2)

    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/node_embeddings_original/0/node_embeddgins_origin_{dim}_{len}_{num}.emb".format(dim=args.dim, len=args.walk_len, num=args.num_walks), "r") as f3:
        all_embeddings = []
        node_names = []
        rows = f3.readlines()
        for row in rows[1:]:
            vector = row.strip().split(" ")[1:]
            all_embeddings.append(vector)
            node_name = row.strip().split(" ")[0]
            node_names.append(node_name)
        all_embeddings_vector = np.array(all_embeddings)
        node_names_vector = np.array(node_names)

    if args.gene_subset == "all":
        with open(args.cancer_gene_data_dir) as f1:
            lines = f1.readlines()
        all_genes_from_msigDB = []
        for line in lines:
            pos_genes = line.split('\t')[2:]
            for each_gene in pos_genes:
                if each_gene in match_dict:
                    all_genes_from_msigDB.append(match_dict[each_gene][0])

        node_names = node_names_vector  # one-to-one match
        labels_for_viz_binary = []

        for node in node_names:
            node_key = list(match_dict_int_ensembl.keys())[list(match_dict_int_ensembl.values()).index(node)]
            if node_key in all_genes_from_msigDB:
                labels_for_viz_binary.append(1)
            else:
                labels_for_viz_binary.append(0)

    else:
        with open(args.cancer_gene_data_dir) as f1:
            lines = f1.readlines()
        msigDB_dict = {}
        for line in lines:
            msigDB_dict[line.split('\t')[0]] = []
            pos_genes = line.split('\t')[2:]
            for each_gene in pos_genes:
                if each_gene in match_dict:
                    msigDB_dict[line.split('\t')[0]].append(match_dict[each_gene][0])

        node_names = node_names_vector  # one-to-one match
        labels_for_viz_binary = []
        for node in node_names:
            node_key = list(match_dict_int_ensembl.keys())[list(match_dict_int_ensembl.values()).index(node)]
            if node_key in msigDB_dict[args.gene_subset]:
                labels_for_viz_binary.append(1)
            else:
                labels_for_viz_binary.append(0)
    
    node_names_int = node_names
    node_names_ensebl = []
    for node in node_names:
        node_names_ensebl.append(list(match_dict_int_ensembl.keys())[list(match_dict_int_ensembl.values()).index(node)])
    
    node_names_entrez = []
    for node2 in node_names_ensebl:
        node_names_entrez.append(list(match_dict_ensembl_entrez_dict.keys())[list(match_dict_ensembl_entrez_dict.values()).index(node2)])

    print(len(node_names_entrez))
    print(len(node_names_ensebl))
    
    
    
    plot_embedding_similarity(all_embeddings_vector, labels_for_viz_binary, args.output_dir, args, node_names_entrez, node_names_ensebl)



if __name__ == "__main__":
    main()
                                    
 