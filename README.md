# Towards Stable Representations for Flexible Protein-Protein Docking


This is the implementation of the paper submitted to NeurIPS 2023 


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
- dataset
- graph_cutoff
- pocket_cutoff

### How to Run and Reproduce the 96 Datasets?

Firstly, specifiy the path of CHEMBL database and the directory to save the data in the configuration
file: `configs/_base_/curators/lbap_defaults.py` for LBAP task  or    `configs/_base_/curators/sbap_defaults.py` for SBAP task.   
The `source_root="YOUR_PATH/chembl_29_sqlite/chembl_29.db"` means the path to the 
chembl29 sqllite file.  The `target_root="data/"` specifies the folder to save the generated data.   

Note that you can download the original chembl29 database with sqllite format from `http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29_sqlite.tar.gz`.


The built-in configuration files are located in:    
`configs/curators/`. Here we provide the 96 config files to __reproduce__ the 96 datasets  in our paper.  Meanwhile, 
you can also customize your own datasets by changing the config files.  

Run `tools/curate.py` to generate dataset. Here are some examples:

Generate datasets for the LBAP task, with `assay` as domain, `core` as noise
level, `IC50` as measurement type, `LBAP` as task type.:

```shell
python tools/curate.py --cfg configs/curators/lbap_core_ic50_assay.py
```

Generate datasets for the SBAP task, with `protein` as domain, `refined` as noise level, `EC50` as
measurement type, `SBAP` as task type.:

```shell
python tools/curate.py --cfg configs/curator/sbap_refined_ec50_protein.py
```

## Benchmarking SOTA OOD Algorithms

Currently we support 6 different baseline algorithms:

- ERM
- IRM
- GroupDro
- Coral
- MixUp
- DANN

Meanwhile, we support various GNN backbones:

- GIN
- GCN
- Weave
- ShcNet
- GAT
- MGCN
- NF
- ATi-FPGNN
- GTransformer

And different backbones for protein sequence modeling:

- Bert
- ProteinBert

### How to Run?

Firstly, run the following command to install.

```shell
python setup.py develop
```

Run the LBAP task with ERM algorithm:

```shell
python tools/train.py configs/algorithms/erm/lbap_core_ec50_assay_erm.py
```                                                        

If you would like to run ERM on other datasets, change the corresponding options inside the above
config file. For example,  `ann_file = 'data/lbap_core_ec50_assay.json'`   specifies the input data.  

Similarly, run the SBAP task with ERM algorithm: 

```shell
python tools/train.py configs/algorithms/erm/sbap_core_ec50_assay_erm.py
``` 


## Reference

:smile:If you find this repo is useful, please consider to cite our paper:

```
@ARTICLE{2022arXiv220109637J,
    author = {{Ji}, Yuanfeng and {Zhang}, Lu and {Wu}, Jiaxiang and {Wu}, Bingzhe and {Huang}, Long-Kai and {Xu}, Tingyang and {Rong}, Yu and {Li}, Lanqing and {Ren}, Jie and {Xue}, Ding and {Lai}, Houtim and {Xu}, Shaoyong and {Feng}, Jing and {Liu}, Wei and {Luo}, Ping and {Zhou}, Shuigeng and {Huang}, Junzhou and {Zhao}, Peilin and {Bian}, Yatao},
    title = "{DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations}",
    journal = {arXiv e-prints},
    keywords = {Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Quantitative Biology - Quantitative Methods},
    year = 2022,
    month = jan,
    eid = {arXiv:2201.09637},
    pages = {arXiv:2201.09637},
    archivePrefix = {arXiv},
    eprint = {2201.09637},
    primaryClass = {cs.LG}
}
```     

## Disclaimer 
This is not an officially supported Tencent product.
