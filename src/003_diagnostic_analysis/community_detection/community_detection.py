from cdlib import algorithms, viz
import networkx as nx
import pickle
import matplotlib.pyplot as plt


G = pickle.load(open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/networkx_graph.pickle', 'rb'))
coms = algorithms.leiden(G)
print(len(coms.communities))
count = 0
for each_com in coms.communities:
    if len(each_com) > 50:
        count += 1

fig = viz.plot_community_graph(G, coms, figsize=(12, 12), top_k=count, node_size=30, plot_overlaps=True, plot_labels=True)
plt.savefig("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/diagnostic_analysis/community_detection/community_detection_top_{count}_coms_community_graph.png".format(count = str(count)))

pos = nx.spring_layout(G)
pickle.dump(pos, open('/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/diagnostic_analysis/community_detection/spring_layout_dict.pickle', 'wb'))
print("yes")

fig = viz.plot_network_clusters(G, coms, pos, figsize=(12, 12), top_k=count, node_size=20, plot_overlaps=True, plot_labels=True)

plt.savefig("/.mounts/labs/reimandlab/private/users/gli/BCB430/Node2Vec/codes/new_implementation_n2v/all_interaction/diagnostic_analysis/community_detection/community_detection_top_{count}_coms_nodes_graph.png".format(count = str(count)))