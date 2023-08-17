from gensim.models import Word2Vec
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load__match_dict(n2v_match_dict_dir):
    with open(n2v_match_dict_dir, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def plot_embedding_viz(node_embeddings_2d, labels, save_path, args):
    node_emb_2d_pos = []
    node_emb_2d_neg = []
    for i in range(len(labels)):
        if labels[i] == 1:
            node_emb_2d_pos.append(node_embeddings_2d[i, :])
        else:
            node_emb_2d_neg.append(node_embeddings_2d[i, :])


    plt.figure(figsize=(10, 8))
    plt.scatter(
      np.array(node_emb_2d_neg)[:, 0],
      np.array(node_emb_2d_neg)[:, 1],
      c = "#a8cce4",
      s = 6,
      alpha = 0.5,
    )
    plt.scatter(
      np.array(node_emb_2d_pos)[:, 0],
      np.array(node_emb_2d_pos)[:, 1],
      c = "#fc6b03",
      s = 6,
      alpha = 0.6,
    )
    plt.xlabel('{}-1'.format(args.dim_reduce_way))
    plt.ylabel('{}-2'.format(args.dim_reduce_way))
    plt.title('{reduce_way} of {gene_subset} dim_{dim} walk_len_{len} num_walks_{num}'.format(reduce_way = args.dim_reduce_way, gene_subset = args.gene_subset, dim=args.dim, len=args.walk_len, num=args.num_walks))
    plt.savefig(save_path + '/embedding_viz_{gene_subset}_{reduce}_{dim}_{walk_len}_{num_walk}.png'.format(gene_subset=args.gene_subset, reduce=args.dim_reduce_way, dim=args.dim, walk_len=args.walk_len, num_walk=args.num_walks))
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n2v_model_emb_dir",
        default=None,
        type=str,
        required=True,
        help="ph.",
    )

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
        "--dim_reduce_way",
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
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all.pkl", 'rb') as f:
        match_dict = pickle.load(f)
        
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all_int_ensembl.pkl", 'rb') as f2:
        match_dict_int_ensembl = pickle.load(f2)
    
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/node_embeddings_original/node_embeddgins_origin.emb", "r") as f3:
        node_names = []
        rows = f3.readlines()
        for row in rows[1:]:
            node_name = row.strip().split(" ")[0]
            node_names.append(node_name)
        node_names_vector = np.array(node_names)
    
    node_embeddings_2d = np.load(args.n2v_model_emb_dir + "/{reduce}_2d_emb_{dim}_{walk_len}_{num_walk}.npy".format(reduce=args.dim_reduce_way, dim=args.dim, walk_len=args.walk_len, num_walk=args.num_walks))
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
                
    
    plot_embedding_viz(node_embeddings_2d, labels_for_viz_binary, args.output_dir, args)
    
if __name__ == "__main__":
    main()