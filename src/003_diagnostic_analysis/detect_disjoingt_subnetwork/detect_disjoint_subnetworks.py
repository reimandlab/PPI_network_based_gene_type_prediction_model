import pickle
import networkx as nx

g = pickle.load(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/networkx_graph.pickle', 'rb'))
components = [g.subgraph(c).copy() for c in nx.connected_components(g)]
print("good")
with open("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/diagnostic_analysis/detect_disjoint_subnetworks/output.txt", "w") as f:
    
    for idx,g in enumerate(components[:1],start=1):
        f.write(f"Component {idx}: Nodes: {g.nodes()} Edges: {g.edges()} \n")