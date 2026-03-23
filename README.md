# SCORE
Semi-supervised Classification Through Representation Learning of EHR Data.

## Description
This folder contains R code for

(1) An R package for fitting SCORE Model (*SCORE*), and

(2) Example code to train the model on simulated dataset (*simulations*).

(3) Example code to run real-world application on synthetic data (*application*).

The implementation of this package was developed with reference to, and partially inspired by, the coding structure of the [PLNmodels R package](http://dx.doi.org/10.1214/18%2DAOAS1177). In particular, aspects of the package organization, model-fitting workflow, and variational inference implementation were adapted to ensure modularity, clarity, and computational efficiency.

We gratefully acknowledge the authors of the PLNmodels package for providing a well-designed framework that facilitated the development of this work.

## Preprint
Wang, L.\*, Li, M.\*, Xia, Z., Liu, M., & Cai, T. (2025). Semi-supervised Clustering Through Representation Learning of Large-scale EHR Data. arXiv preprint arXiv:2505.20731. [link](https://arxiv.org/pdf/2505.20731)

## References
J. Chiquet, M. Mariadassou and S. Robin: Variational inference for probabilistic Poisson PCA, the Annals of Applied Statistics, 12: 2674–2698, 2018. [link](http://dx.doi.org/10.1214/18%2DAOAS1177)

R package: https://github.com/PLN-team/PLNmodels/tree/master

