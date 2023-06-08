# Towards Stable Representations for Flexible Protein-Protein Docking


This is the implementation of the paper **FlexDock** submitted to NeurIPS 2023 


## Dependencies

FlexDock needs the following environment: 

```shell
python==3.7
numpy==1.22.4
torch-geometric==2.2.0
cuda==10.2
torch==1.11.0
dgl==0.8.1
biopandas==0.4.1
dgllife==0.2.9
joblib==1.1.0
prody==2.4.0
```   
For convenience, you can follow 2 steps:
1. Run ```conda env create -f env.yaml```
2. Download the following whl files to `./file/`: [torch-scatter](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl), [torch-sparse](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl), [torch-cluster](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl), [torch-spline-conv](https://data.pyg.org/whl/torch-1.11.0%2Bcu102/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl).

```
cd ./file
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
```

## Dataset Curation

First, you need to generate the required graph structured data for complex with our code. The curator includes two datasets:

- Docking Benchmark 5.5 (DB5.5).
- Database of Interacting Protein Structures (DIPS).

For data preparations, you can choose the configuration as follows:
- '**dataset**'. \["dips","db55"\]: Datasets will be processed separately, so please choose one.
- '**graph_cutoff**'. If the physical distance between two residues in a protein is less than this value, they will be assigned an edge in the KNN graph.
- '**graph_max_neighbor**'. It means
- '**pocket_cutoff**'. If the physical distance between inter-protein residues is less than this value, they will be considered in the pocket.
- '**prody**'. \[True, False\]: "True" means using the ProDy software to simulate the unbound protein conformation changes, otherwise Gaussian noise will be used to represent the flexibility.
- '**data_frac**'. You can set a number between 0 and 1 to handle a portion of the DIPS dataset (as it is relatively large).

You can preprocess the raw data as follows for DB5.5:
```
python src.preprocess_raw_data.py -dataset db55 -graph_cutoff 20 -graph_max_neighbor 10 -pocket_cutoff 8 -prody True
```

## How to run

You can find a detailed explanation of the parameters in ```./src/utils/args.py```.

By setting the 'toy' parameter to True, you can successfully train a toy example. This means that the FlexDock model will be validated and tested on DB5.5 without pretraining on DIPS.
```
python -m src.train -toy True -tune False -h_dim True 32 -atten_head 8 -SEGCN_layer 3 -dropout 0.2 --gamma 0.2 
```
To reproduce the results in the paper, you can run the following in sequence.
```
python -m src.train -data dips -tune False
```
```
python -m src.train -data db55 -tune True
```
