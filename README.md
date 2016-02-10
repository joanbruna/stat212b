# stat212b
Topics Course on Deep Learning for Spring 2016

UC Berkeley, Statistics Department

##Syllabus

### 1st part: Convolutional Neural Networks
  - Invariance, stability.
  - Variability models (deformation model, stochastic model). 
  - Scattering
  - Extensions 
  - Group Formalism 
  - Supervised Learning: classification. 
  - Properties of CNN representations: invertibility, stability, invariance. 
  - covariance/invariance: capsules and related models.
  - Connections with other models: dictionary learning, LISTA, Random Forests.
  - Other tasks: localization, regression. 
  - Embeddings (DrLim), inverse problems 
  - Extensions to non-euclidean domains.
  - Dynamical systems: RNNs and optimal control. 
  - Guest Lecture: Wojciech Zaremba (OpenAI)
  
### 2nd part: Deep Unsupervised Learning
 - Autoencoders (standard, denoising, contractive, etc.)
 - Variational Autoencoders
 - Adversarial Generative Networks
 - Maximum Entropy Distributions
 - Open Problems
 - Guest Lecture: Ian Goodfellow (Google)

### 3rd part: Miscellaneous Topics
- Non-convex optimization theory for deep networks 
- Stochastic Optimization
- Attention and Memory Models
- Guest Lecture: Yann Dauphin (Facebook AI Research)
  


## Schedule

Lec1 Jan 19: Intro and Logistics

Lec2 Jan 21: Representations for Recognition : stability, variability. 
 Kernel approaches / Feature extraction. Properties. 

  *recommended reading*
  -  [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.
  -  [Elements of Statistical Learning, chapt. 12](http://statweb.stanford.edu/~tibs/ElemStatLearn/), Hastie, Tibshirani, Friedman.
 
Lec3 Jan 26: Groups, Invariants and Filters.

  *recommended reading*
  - [Learning Stable Group Invariant Representations with Convolutional Networks](http://cims.nyu.edu/~bruna/Misc/iclr_group2.pdf)
  - [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.
  - [A Wavelet Tour of Signal Processing, chapt 2-5,7](https://www.ceremade.dauphine.fr/~peyre/wavelet-tour/), S. Mallat.

Lec4 Jan 28: Scattering Convolutional Networks.

  *recommended reading*
  - [Invariant Scattering Convolutional Networks](http://arxiv.org/pdf/1203.1513v2.pdf)
 
   *further reading*
  - [Group Invariant Scattering](http://arxiv.org/abs/1101.2286), S. Mallat
  - [Scattering Representations for Recognition](http://cims.nyu.edu/~bruna/PhD.html)

Lec5 Feb 2: Further Scattering: Properties and Extensions.

  *recommended reading*
  - [Rotation, Scaling and Deformation Invariant Scattering for Texture Discrimination](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sifre_Rotation_Scaling_and_2013_CVPR_paper.pdf), Sifre & Mallat.

Lec6 Feb 4: Convolutional Neural Networks: Geometry and first Properties.

  *recommended reading*
  - [Deep Learning](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) Y. LeCun, Bengio & Hinton.
  - [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.

Lec7 Feb 9: Properties of learnt CNN representations: Covariance and Invariance, redundancy, invertibility

  *recommended reading*
  - [Deep Neural Networks with Random Gaussian Weights: A universal Classification Strategy?](http://arxiv.org/abs/1504.08291), R. Giryes, G. Sapiro, A. Bronstein.
  - [Intriguing Properties of Neural Networks](http://arxiv.org/abs/1312.6199) C. Szegedy et al. 
  - [Geodesics of Learnt Representations](http://arxiv.org/abs/1511.06394) O. Henaff & E. Simoncelli.
  - [Inverting Visual Representations with Convolutional Networks](http://arxiv.org/abs/1506.02753), A. Dosovitskiy, T. Brox.
  - [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901) M. Zeiler, R. Fergus.

Lec8 Feb 11: Connections with other models (DL, Lista, Random Forests, CART)

Lec9 Feb 16: Representations of stationary processes. Properties. 

Lec10 Feb 18: Other high level tasks: localization, regression, embedding, inverse problems. 

Lec11 Feb 23: Extensions to non-Euclidean domain. Sequential Data RNNs. 

Lec12 Feb 25: Guest Lecture ( W. Zaremba, OpenAI ) 

Lec13 Mar 1: Unsupervised Learning: autoencoders. Density estimation. Parzen estimators. Curse of dimensionality

Lec14 Mar 3: Variational Autoencoders

Lec15 Mar 8: Adversarial Generative Networks

Lec16 Mar 10: Maximum Entropy Distributions

Lec17 Mar 29: Self-supervised models (analogies, video prediction, text, word2vec). 

Lec18 Mar 31: Guest Lecture ( I. Goodfellow, Google Brain ) 

Lec19 Apr 5: Non-convex Optimization: parameter redundancy, spin-glass, optimiality certificates. stability

Lec20 Apr 7: Tensor Decompositions

Lec20 Apr 12: Stochastic Optimization, Batch Normalization, Dropout

Lec21 Apr 14: Reasoning, Attention and Memory: New trends of the field and challenges. 
      limits of sequential representations (need for attention and memory). 
      modern enhancements (NTM, Memnets, Stack/RNNs, etc.)

Lec22 Apr 19: Guest Lecture (Y. Dauphin, Facebook AI Research)

Lec23-25 Oral Presentations
