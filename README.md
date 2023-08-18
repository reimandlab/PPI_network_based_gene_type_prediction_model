# PPI_network_based_gene_type_prediction_model
This is the repository for Reagan (Gen) LI's BCB430 summer project.

The BCB430 summer project aims to construct a machine learning model that can classify genes into cancer genes and non-cancer genes according to the structure information extracted from protein-protein interaction (PPI) network data.

The code for the project can be found in src, the steps of the project are listed in order of 001 to 006 in the src folder.


001 is used for the pre-processing of the PPI data.  
002 is used to construct a graph according to the PPI data.  
003 is for some diagnostic analysis on the graph generated from PPI data.  
004 are used for generation of node embedding features in different dimensions.  
005 is used for visualization of the embedding features in plots.  
006 is used to train, validate and test the classfier using generated features.


Note: src/node2vec are not wrtten by Gen LI. it is an implementation from Aditya Grover and Jure Leskovec, whose ciation is shown below:

@inproceedings{node2vec-kdd2016,  
author = {Grover, Aditya and Leskovec, Jure},  
 title = {node2vec: Scalable Feature Learning for Networks},  
 booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},  
 year = {2016}  
}



All source data can be found in data folder. Because of the large size of the PPI data, it is stored on Google drive, please download from the following link: https://drive.google.com/drive/folders/1z1kHReV_lx1qarT5XAzcRUWOjR_povE7?usp=share_link//
