# scSemiAAE--A semi-supervised clustering model for single-cell RNA-seq data
Single-cell RNA sequencing (scRNA-seq) strives to capture cellular diversity with higher resolution than bulk RNA sequencing. Clustering analysis is a crucial step as it provides an opportunity to further identify and uncover undiscovered cell types. Most existing clustering methods support unsupervised clustering but cannot integrate prior information.When faced with the high dimensionality of scRNA-Seq data and common dropout events, purely unsupervised clustering methods may fail to produce biologically interpretable clusters, which complicates cell type assignment. Here, we propose scSemiAAE, a semi-supervised clustering model for scRNA sequence analysis using deep generative neural networks. Specifically, scSemiAAE carefully designs a ZINB loss-based autoencoder architecture that inherently integrates adversarial training and semi-supervised modules in the latent space. In a series of experiments on scRNA-seq datasets spanning thousands to tens of thousands of cells, scSemiAAE can significantly improve clustering performance compared to dozens of unsupervised and semi-supervised clustering algorithms, promoting clustering and interpretability of downstream analyses.

Requirements:

Python --- 3.8.13

pytorch -- 1.11.0

Scanpy --- 1.0.4

Nvidia Tesla P40

Files:

scSemiAAE.py -- implementation of scSemiAAE algorithm

layers.py -- Definition of the zero-inflated negative binomial distribution

filter.py -- Load and process the data

Branch:

Data -- Data used in the paper
