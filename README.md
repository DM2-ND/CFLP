Counterfactual Graph Learning for Link Prediction
====
This repository contains the source code for the paper:

[Counterfactual Graph Learning for Link Prediction](https://arxiv.org/pdf/2106.02172.pdf)

by by [Tong Zhao](https://tzhao.io/) (tzhao2@nd.edu), Gang Liu, Daheng Wang, Wenhao Yu, and [Meng Jiang](http://www.meng-jiang.com/).

## Requirements

This code package was developed and tested with Python 3.8.5 and PyTorch 1.6.0. All dependencies specified in the ```requirements.txt``` file. The packages can be installed by
```
pip install -r requirements.txt
```

## Usage
The step of finding all the counterfactual links can be slow for the first run, please adjust the ```--n_workers``` parameter according to available processes if you are trying out different settings. The cached files for the counterfactual links that were used in our experiments can be found [here](https://www.dropbox.com/sh/zumzy19mdm57yw8/AAC6m8-PQDT-ygbEvByDlOcna?dl=0), please download and put them under ```data/T_files/``` before reproducing our experiments.

Following are the commands to reproduce our experiment results on different datasets.
```
# Cora
python main.py --dataset cora --metric auc --alpha 1 --beta 1 --gamma 30 --lr 0.1 --embraw mvgrl --t kcore --neg_rate 50 --jk_mode mean --trail 20

# CiteSeer
python main.py --dataset citeseer --metric auc --alpha 1 --beta 1 --gamma 30 --lr=0.1 --embraw mvgrl --t kcore --neg_rate 50 --jk_mode mean --trail 20

# PubMed
python main.py --dataset pubmed --metric auc --alpha 1 --beta 1 --gamma 30 --lr 0.1 --embraw mvgrl --t kcore --neg_rate 40 --jk_mode mean --batch_size 12000 --epochs 200 --patience 50 --trail 20

# Facebook
python main.py --dataset facebook --metric hits@20 --alpha 1e-3 --beta 1e-3 --gamma 30 --lr 0.005 --embraw mvgrl --t louvain --neg_rate 1 --jk_mode mean --trail 20

# OGBL-ddi
python main.py --dataset ogbl-ddi --metric hits@20 --alpha 1e-3 --beta 1e-3 --gamma 10 --lr 0.01 --embraw mvgrl --t louvain  --neg_rate 1 --jk_mode mean --epochs=200 --epochs_ft=200 --patience=50 --trail 20
```

## Cite
If you find this repository useful in your research, please cite our paper:

```bibtex
@article{zhao2021counterfactual,
  title={Counterfactual Graph Learning for Link Prediction},
  author={Zhao, Tong and Liu, Gang and Wang, Daheng and Yu, Wenhao and Jiang, Meng},
  journal={arXiv preprint arXiv:2106.02172},
  year={2021}
}
```

