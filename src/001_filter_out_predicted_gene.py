import csv
from tqdm import tqdm


def search_and_return(data_path):
    match_file = csv.reader(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/ppi_data/ensembl_hg38-symbols-tcga_entrez.tsv', "r"), delimiter="\t")
    match_dict = {}
    next(match_file)
    for row in match_file:
        if row[-1] != '':
            match_dict[str(int(float(row[-1])))] = row[0]
    print(len(match_dict.keys()))
     
    protein_coding_file = csv.reader(open("/.mounts/labs/reimandlab/private/users/gli/BCB430/ppi_data/hg38_protein_coding.tsv", 'r'), delimiter="\t")
    protein_coding_gene_dict = {}
    next(protein_coding_file)
    for row in protein_coding_file:
        protein_coding_gene_dict[row[0]] = " "
    
    with open(data_path) as f:
        interaction_list = []
        lines = f.readlines()
        node_list = []
        edge_list = []
        for line in tqdm(lines[1:]):
            if line.split("\t")[0].split(":")[-1] in match_dict and line.split("\t")[1].split(":")[-1] in match_dict:
                ensembl_ID_A = match_dict[line.split("\t")[0].split(":")[-1]][0]
                ensembl_ID_B = match_dict[line.split("\t")[1].split(":")[-1]][0]
                if ensembl_ID_A in protein_coding_gene_dict and ensembl_ID_B in protein_coding_gene_dict: # make sure that both proteins in an interaction have match ENSG ID and are protein-coding genes
                    node_list.append(ensembl_ID_A)
                    node_list.append(ensembl_ID_B)
                    edge_list.append((ensembl_ID_A, ensembl_ID_B))
        
    node_list_removed = list(set(node_list))
    return node_list_removed, edge_list, match_dict
                
if __name__ == "__main__":
    search_and_return()