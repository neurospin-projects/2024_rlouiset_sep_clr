# SepCLR

This repository contains the script for the ICLR 2024 accepted paper.
Paper title: SEPARATING COMMON FROM SALIENT PATTERNS WITH CONTRASTIVE REPRESENTATION LEARNING
Authors: Robin Louiset, Edouard Duchesnay, Antoine Grigis, Pietro Gori

You will need a python 3.8 interpreter to run the scripts.
Datasets can be downloaded on links referenced in the paper's main text.

You can lauch a training script by simply writing:
python cifar_mnist/sep_clr_k-jem.py

And create a dataset (for celeba dataset, it's actually in two parts) with:
python celeba_accessories/create_dataset.py

Please cite this paper if you use a dataset, idea, or loss (k-JEM, s'-uniformity or information-less reg for ex.)

If you are having questions about the losses, the intuitions or the maths, please feel free to contact me at:
robin.louiset@gmail.com