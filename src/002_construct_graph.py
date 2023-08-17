from filter_out_predicted_gene import search_and_return
import networkx as nx
import pickle

def construct_graph(file_path, interaction):
    node_list, edge_list, match_dict = search_and_return(file_path)
    print("Graph nodes num: ", len(node_list))
    print("Graph edges num: ", len(edge_list))
    
    match_dict_int_ensembl = {}
    count = 0
    for edge in edge_list:
        if edge[0] not in match_dict_int_ensembl:
            match_dict_int_ensembl[edge[0]] = str(count)
            count += 1
        if edge[1] not in match_dict_int_ensembl:
            match_dict_int_ensembl[edge[1]] = str(count)
            count += 1
    
    
    with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/graph_{interaction}.edgelist".format(interaction=interaction), "w") as f1:
        for edge in edge_list:
            f1.write("{node1} {node2}\n".format(node1=match_dict_int_ensembl[edge[0]], node2=match_dict_int_ensembl[edge[1]]))
    
    with open('/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_{interaction}_int_ensembl.pkl'.format(interaction=interaction), 'wb') as f:
        pickle.dump(match_dict_int_ensembl, f)
    
    with open('/.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/match_dict_{interaction}_one_to_one.pkl'.format(interaction=interaction), 'wb') as f:
        pickle.dump(match_dict, f)
    
    

    
if __name__ == "__main__":
    construct_graph("/.mounts/labs/reimandlab/private/users/gli/BCB430/ppi_data/BIOGRID-ORGANISM-Homo_sapiens-4.4.221.mitab.txt", "all")