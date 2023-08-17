from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
import argparse
from umap import UMAP
import pickle
import os
import numpy as np


def load_n2v_match_dict(n2v_match_dict_dir):
    with open(n2v_match_dict_dir, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def visualize_embedding_tsne(vectors, save_path, args):
    
    node_embeddings = (
    vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    
    tsne = TSNE(n_components=2, random_state = 42)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    np.save(save_path + "/tsne_2d_emb_{dim}_{walk_len}_{num_walk}.npy".format(dim=args.dim, walk_len=args.walk_len, num_walk=args.num_walks), node_embeddings_2d)
    

def visualize_embedding_umap(vectors, save_path, args):
    node_embeddings = (
    vectors
    )
    
    umap_2d = UMAP(n_components=2, random_state = 42) #, metric="hamming"
    #print("node_embeddings: ", node_embeddings)
    print(node_embeddings.shape)
    node_embeddings_2d = umap_2d.fit_transform(node_embeddings)
    np.save(save_path + "/umap_2d_emb_{dim}_{walk_len}_{num_walk}.npy".format(dim=args.dim, walk_len=args.walk_len, num_walk=args.num_walks), node_embeddings_2d)
    

def visualize_embedding_pca(vectors, save_path, args):
    node_embeddings = (
    vectors
    )
    pca = PCA(n_components = 2, random_state = 42)
    node_embeddings_2d = pca.fit_transform(node_embeddings)
    np.save(save_path + "/pca_2d_emb_{dim}_{walk_len}_{num_walk}.npy".format(dim=args.dim, walk_len=args.walk_len, num_walk=args.num_walks), node_embeddings_2d)
    

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
        "--dim",
        default=None,
        type=str,
        required=True,
        help="Path where the output files are saved.",
    )
    parser.add_argument(
        "--walk_len",
        default=None,
        type=str,
        required=True,
        help="Path where the output files are saved.",
    )
    parser.add_argument(
        "--num_walks",
        default=None,
        type=str,
        required=True,
        help="Path where the output files are saved.",
    )


    args = parser.parse_args()             
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/node_embeddings_original/node_embeddgins_origin_{dim}_{walk_len}_{num_walks}.emb".format(dim=args.dim, walk_len=args.walk_len, num_walks=args.num_walks), "r") as f:
        all_embeddings = []
        # node_names = []
        rows = f.readlines()
        for row in rows[1:]:
            vector = row.strip().split(" ")[1:]
            all_embeddings.append(vector)
        all_embeddings_vector = np.array(all_embeddings)
        visualize_embedding_umap(all_embeddings_vector, args.output_dir, args)
        visualize_embedding_tsne(all_embeddings_vector, args.output_dir, args)
        visualize_embedding_pca(all_embeddings_vector, args.output_dir, args)

if __name__ == "__main__":
    main()
                
    
    
    




