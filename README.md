# [MIDL 2024] Self-supervised pretraining in the wild imparts image acquisition robustness to medical image transformers: an application to lung cancer segmentation
## Jue Jiang and Harini Veeraraghavan

### Analysis details
This repository provides the code and models associated with the our MIDL 2024 paper [Self-supervised pretraining in the wild imparts image acquisition robustness to medical image transformers: an application to lung cancer segmentation](https://openreview.net/pdf?id=G9Te2IevNm). 

Self-supervised learning (SSL) is an approach to extract universally reusable feature representations without requiring expert annotations. Increasing availability of large scale medical image datasets has now made it possible to create medical image foundation models using SSL applied to loosely curated and unlabeled medical datasets that are unrelated to the downstream tasks. Models created by pretraining with such large-scale datasets or "wild" pretraining differ from typical approaches that employ self-pretraining where SSL is applied on the same downstream task dataset prior to supervised training in an effort to extract relevant feature representations. Medical images are highly heterogeneous and exhibit wide variations due to imaging acquisition differences. In this context, understanding the benefits of "wild" versus "self" pretraining for enhancing the robustness to imaging variations is important and is the focus of our study. 

# Key results from this work

1. Wild-pretraining of transformer networks improves robustness to CT imaging acquisitions. In particular, Swin transformer created using wild-pretraining was significantly more accurate than its self-pretrained counterpart and also showed higher robustness to CT imaging differences.
2. 

Please add details
# Requirements

# Installaion




