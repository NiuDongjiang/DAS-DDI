# DAS-DDI: A Dual-View Framework with Drug Association and Drug Structure for Drug-Drug Interaction Prediction

## Virtual environment

torch==1.9.0

torch-geometric==2.2.0

torch-sparse==0.6.12

python==3.7

dgl==0.6.1 

numpy==1.20.0 

pandas==1.3.5

## Dataset

DrugBank: [SA-DDI/drugbank/data at dev · guaguabujianle/SA-DDI (github.com)](https://github.com/guaguabujianle/SA-DDI/tree/dev/drugbank/data)

ChChMiner: http://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html

ZhangDDI: https://github.com/zw9977129/drug-drug-interaction/tree/master/dataset

## Useage

If you want to evaluate different datasets, please change the 'dataset_name' in args.json.

### Train the model

python transductive_train.py

