library(Seurat)
# 1. Load data and perform standard preprocessing steps
pbmc.data <- Read10X(data.dir = "path/to/data") # Load 10x data
pbmc <- CreateSeuratObject(counts = pbmc.data) # Create Seurat object
pbmc <- NormalizeData(pbmc) # Normalize data
pbmc <- FindVariableFeatures(pbmc) # Identify variable genes
pbmc <- ScaleData(pbmc) # Scale data
pbmc <- RunPCA(pbmc, npcs = 30) # Perform PCA
pbmc <- FindNeighbors(pbmc) # Find nearest neighbors
pbmc <- FindClusters(pbmc) # Perform clustering
# 2. Read custom clustering results from csv file
custom_clusters <- read.csv("path/to/custom_clusters.csv", header = TRUE, stringsAsFactors = FALSE) # Read csv file into R
# 3. Set custom cluster results
Idents(pbmc) <- custom_clusters$ClusterID # Assign custom clusters to Seurat object
# 4. Visualize custom cluster results
DimPlot(pbmc, reduction = "pca", group.by = "ClusterID") # Visualize custom clusters

##In the above code example, we assume that the name of the cluster identifier column is "ClusterID".
And it is worth noting that the seurat package requires the input of a 10X file, which needs to include separate cell name files and gene name files. 
Similarly, for the Scanpy platform, we can still follow the above idea.
