# ATLAS Graph - Link Prediction
Link to the Task Presentation: https://drive.google.com/drive/folders/1wVFBRVk_FMRwD3y3sknhjwwEPrJisk4D?usp=sharing 

This repo has code and resources to perform link prediction on the ATLAS graph dataset. Graph Analysis and plotting is done using gephi.
The project involves masking some edges and training models using:

1. **Node2Vec**
2. **Graph Neural Networks (GNNs)** with PyTorch Geometric (PyG)

## Dataset
The ATLAS graph dataset is a directed network with last letter of A =  first letter of B edge connection. 
The dataset is preprocessed to generate node embeddings and train models for link prediction.

## Features
- **Node2Vec for Unsupervised Learning**: Generates node embeddings based on random walks.
- **GNN-based Link Prediction**: Uses PyTorch Geometric for training a graph neural network to predict missing edges.
- **Edge Masking for Evaluation**: A subset of edges is hidden during training for model validation.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install torch torch-geometric networkx numpy scikit-learn tqdm
```

## Usage
### 1. Dataset Creation
Run the preprocessing script to prepare the graph.
```bash
python3 country_graph.py city_graph.py final.py
```

### 2. Link Prediction Analysis
Train Node2Vec and PyG by running:
```bash
python bonus.py
```

## Results
The models are evaluated based on link prediction accuracy, using metrics such as:
- Mean
- Standard Deviation
for node2vec, and
- Area Under the Curve (AUC)
- Precision-Recall Score
for PyG

### Author
Sudhanva Joshi

