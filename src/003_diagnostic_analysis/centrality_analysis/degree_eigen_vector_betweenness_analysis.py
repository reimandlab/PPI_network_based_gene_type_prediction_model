from tqdm import tqdm
import pickle
import networkx as nx
import csv
import argparse

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
    


    args = parser.parse_args()
    # load graph object from file
    G = pickle.load(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/networkx_graph.pickle', 'rb'))

    degree_dict = nx.degree_centrality(G) #int: value
    eigen_vector_dict = nx.eigenvector_centrality(G) #int: value
    betweeness_dict = nx.betweenness_centrality(G, k = 100)#int: value
    
    with open(args.cancer_gene_data_dir) as f1:
            lines = f1.readlines()
            msigDB_dict = {}
            for line in lines:
                msigDB_dict[line.split('\t')[0]] = []
                pos_genes = line.split('\t')[2:]
                for each_gene in pos_genes:
                    msigDB_dict[line.split('\t')[0]].append(each_gene.strip()) #hallmark_name: entrez


    match_dict_int_entrez = pickle.load(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_all_interaction_int_entrez.pkl', 'rb'))                   



    if args.gene_subset == "all":
        pos_sample_feature = []
        neg_sample_feature = []
        for i in range(17931):
            if match_dict_int_entrez[str(i)] in sum(msigDB_dict.values(), []):
                pos_sample_feature.append([float(degree_dict[str(i)]), float(eigen_vector_dict[str(i)]), float(betweeness_dict[str(i)]), "pos"])
            else:
                neg_sample_feature.append([float(degree_dict[str(i)]), float(eigen_vector_dict[str(i)]), float(betweeness_dict[str(i)]), "neg"])

        
                           
    else:
        pos_sample_feature = []
        neg_sample_feature = []
        for i in range(17931):
            if match_dict_int_entrez[str(i)] in msigDB_dict[args.gene_subset]:
                pos_sample_feature.append([float(degree_dict[str(i)]), float(eigen_vector_dict[str(i)]), float(betweeness_dict[str(i)]), "pos"])
            else:
                neg_sample_feature.append([float(degree_dict[str(i)]), float(eigen_vector_dict[str(i)]), float(betweeness_dict[str(i)]), "neg"])

    with open(args.output_dir + "/{set}_all_in_one_analysis_normalized.csv".format(set=args.gene_subset), 'w') as f:
        writer = csv.writer(f)

        # write the header
        header = ["degree", "eigen_vector", "betweeness","label"]
        writer.writerow(header)
        writer.writerows(pos_sample_feature)
        writer.writerows(neg_sample_feature)
        
    
    
if __name__ == "__main__":
    main()