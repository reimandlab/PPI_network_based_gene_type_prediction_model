#!/bin/bash
for round in 0 1 2 3 4
do
  python /.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/node2vec/src/main.py --input /.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/graph_and_match_dict/graph_all.edgelist --output /.mounts/labs/reimandlab/private/users/gli/BCB430/codes/new_implementation_n2v/all_interaction/node_embeddings_original/$round/node_embeddgins_origin_128_40_20.emb --dimensions 128 --walk-length 40 --num-walks 20 --window-size 1
done
echo 'Finished!'
