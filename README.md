# stat212b
Topics Course on Deep Learning for Spring 2016

by Joan Bruna, UC Berkeley, Statistics Department

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
  - Guest Lecture: **Wojciech Zaremba** (OpenAI)
  
### 2nd part: Deep Unsupervised Learning
 - Autoencoders (standard, denoising, contractive, etc.)
 - Variational Autoencoders
 - Adversarial Generative Networks
 - Maximum Entropy Distributions
 - Open Problems
 - Guest Lecture: Soumith Chintala (Facebook AI Research)

### 3rd part: Miscellaneous Topics
- Non-convex optimization theory for deep networks 
- Stochastic Optimization
- Attention and Memory Models
- Guest Lecture: **Yann Dauphin** (Facebook AI Research)
  


## Schedule

- **[Lec1](lec1.pdf)** Jan 19: Intro and Logistics

- **[Lec2](lec2.pdf)** Jan 21: Representations for Recognition : stability, variability. 
 Kernel approaches / Feature extraction. Properties. 

  *recommended reading:*
  - [Elements of Statistical Learning, chapt. 12](http://statweb.stanford.edu/~tibs/ElemStatLearn/), Hastie, Tibshirani, Friedman.
  - [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.
 
- **[Lec3](lec3.pdf)** Jan 26: Groups, Invariants and Filters.

  *recommended reading*
  - [Learning Stable Group Invariant Representations with Convolutional Networks](http://cims.nyu.edu/~bruna/Misc/iclr_group2.pdf)
  - [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.
  - [A Wavelet Tour of Signal Processing, chapt 2-5,7](https://www.ceremade.dauphine.fr/~peyre/wavelet-tour/), S. Mallat.

- **[Lec4](lec4.pdf)** Jan 28: Scattering Convolutional Networks.

  *recommended reading*
  - [Invariant Scattering Convolutional Networks](http://arxiv.org/pdf/1203.1513v2.pdf)
 
   *further reading*
  - [Group Invariant Scattering](http://arxiv.org/abs/1101.2286), S. Mallat
  - [Scattering Representations for Recognition](http://cims.nyu.edu/~bruna/PhD.html)

- **[Lec5](lec5.pdf)** Feb 2: Further Scattering: Properties and Extensions.

  *recommended reading*
  - [Rotation, Scaling and Deformation Invariant Scattering for Texture Discrimination](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sifre_Rotation_Scaling_and_2013_CVPR_paper.pdf), Sifre & Mallat.

- **[Lec6](lec6.pdf)** Feb 4: Convolutional Neural Networks: Geometry and first Properties.

  *recommended reading*
  - [Deep Learning](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) Y. LeCun, Bengio & Hinton.
  - [Understanding Deep Convolutional Networks](http://arxiv.org/pdf/1601.04920.pdf), S. Mallat.

- **[Lec7](lec7.pdf)** Feb 9: Properties of learnt CNN representations: Covariance and Invariance, redundancy, invertibility

  *recommended reading*
  - [Deep Neural Networks with Random Gaussian Weights: A universal Classification Strategy?](http://arxiv.org/abs/1504.08291), R. Giryes, G. Sapiro, A. Bronstein.
  - [Intriguing Properties of Neural Networks](http://arxiv.org/abs/1312.6199) C. Szegedy et al. 
  - [Geodesics of Learnt Representations](http://arxiv.org/abs/1511.06394) O. Henaff & E. Simoncelli.
  - [Inverting Visual Representations with Convolutional Networks](http://arxiv.org/abs/1506.02753), A. Dosovitskiy, T. Brox.
  - [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901) M. Zeiler, R. Fergus.

- **[Lec8](lec8.pdf)** Feb 11: Connections with other models (DL, Lista, Random Forests, CART) 

 *recommended reading*
 - [Proximal Splitting Methods in Signal Processing](http://arxiv.org/pdf/0912.3522v4.pdf) Combettes & Pesquet.
 - [A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems](http://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf) Beck & Teboulle
 - [Learning Fast Approximations of Sparse Coding](http://www.cs.nyu.edu/~kgregor/gregor-icml-10.pdf) K. Gregor & Y. LeCun
 - [Task Driven Dictionary Learning](http://arxiv.org/pdf/1009.5358.pdf) J. Mairal, F. Bach, J. Ponce
 - [Exploiting Generative Models in Discriminative Classifiers](http://papers.nips.cc/paper/1520-exploiting-generative-models-in-discriminative-classifiers.pdf) T. Jaakkola & D. Haussler
 - [Improving the Fisher Kernel for Large-Scale Image Classification](https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf) F. Perronnin et al.
 - [NetVLAD](http://www.di.ens.fr/willow/research/netvlad/) R. Arandjelovic et al.

- **[Lec9](lec9.pdf)** Feb 16: Other high level tasks: localization, regression, embedding, inverse problems. 

 *recommended reading*
 - [Object Detection with Discriminatively Trained Deformable Parts Model](https://www.cs.berkeley.edu/~rbg/papers/Object-Detection-with-Discriminatively-Trained-Part-Based-Models--Felzenszwalb-Girshick-McAllester-Ramanan.pdf) Felzenswalb, Girshick, McAllester and Ramanan, PAMI'10
 - [Deformable Parts Models are Convolutional Neural Networks](http://arxiv.org/abs/1409.5403), Girshick, Iandola, Darrel and Malik, CVPR'15.
 - [Rich Feature Hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) Girshick, Donahue, Darrel and Malik, PAMI'14.
 - [Graphical Models, message-passing algorithms and convex optimization](http://www.eecs.berkeley.edu/~wainwrig/Talks/A_GraphModel_Tutorial) M. Wainwright.
 - [Conditional Random Fields as Recurrent Neural Networks](http://arxiv.org/pdf/1502.03240.pdf) Zheng et al, ICCV'15
 - [Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation](http://arxiv.org/abs/1406.2984) Tompson, Jain, LeCun and Bregler, NIPS'14.

- **[Lec10](lec10.pdf)** Feb 18:  Extensions to non-Euclidean domain. Representations of stationary processes. Properties. 

 *recommended reading*
 - [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) Hadsell, Chopra, LeCun,'06. 
 - [Deep Metric Learning via Lifted Structured Feature Embedding](http://arxiv.org/abs/1511.06452) Oh Song, Xiang, Jegelka, Savarese,'15.
 - [Spectral Networks and Locally Connected Networks on Graphs](http://arxiv.org/abs/1312.6203) Bruna, Szlam, Zaremba, LeCun,'14.
 - [Spatial Transformer Networks](http://arxiv.org/abs/1506.02025) Jaderberg, Simonyan, Zisserman, Kavukcuoglu,'15.
 - [Intermittent Process Analysis with Scattering Moments](http://arxiv.org/abs/1311.4104) Bruna, Mallat, Bacry, Muzy,'14.

- **[Lec11](lec11_guest_wojciechzaremba.pdf)** Feb 23: Guest Lecture ( W. Zaremba, OpenAI )  Discrete Neural Turing Machines. 

- **[Lec12](lec12.pdf)** Feb 25: Representations of Stationary Processes (contd). Sequential Data: Recurrent Neural Networks.

  *recommended reading*
  - [Intermittent Process Analysis with Scattering Moments](http://arxiv.org/abs/1311.4104) J.B., Mallat, Bacry and Muzy, Annals of Statistics,'13. 
  - [A mathematical motivation for complex-valued convolutional networks](http://arxiv.org/abs/1503.03438) Tygert et al., Neural Computation'16. 
  - [Texture Synthesis Using Convolutional Neural Networks](http://arxiv.org/abs/1505.07376) Gatys, Ecker, Betghe, NIPS'15.
  - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576), Gatys, Ecker, Betghe, '15.
  - [Time Series Analysis and its Applications](http://www.stat.pitt.edu/stoffer/tsa3/) Shumway, Stoffer, Chapter 6.
  - [Deep Learning](http://www.deeplearningbook.org) Goodfellow, Bengio, Courville,'16. Chapter 10. 

- **[Lec13](lec13.pdf)** Mar 1: Recurrent Neural Networks (contd). Long Short Term Memory. Applications. 

 *recommended reading*
 - [Deep Learning](http://www.deeplearningbook.org) Goodfellow, Bengio, Courville,'16. Chapter 10. 
 - [Generating Sequences with Recurrent Neural Networks](http://arxiv.org/abs/1308.0850) A. Graves. 
 - [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) A. Karpathy
 - [The Unreasonable effectiveness of Character-level Language Models](http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139) Y. Goldberg

- **[Lec14](lec14.pdf)** Mar 3: Unsupervised Learning: Curse of dimensionality, Density estimation. Graphical Models, Latent Variable models.

  *recommended reading*
  - [Describing Multimedia Content Using Attention-based Encoder-Decoder Networks](http://arxiv.org/pdf/1507.01053.pdf) K. Cho, A. Courville, Y. Bengio
  - [Graphical Models, Exponential Families and Variational Inference](https://www.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) M. Wainwright, M. Jordan.
  

- **[Lec15](lec15.pdf)** Mar 8: Autoencoders. Variational Inference. Variational Autoencoders. 

  *recommended reading*
  - [Graphical Models, Exponential Families and Variational Inference, chapter 3](https://www.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) M. Wainwright, M. Jordan.
  - [Variational Inference with Stochastic Search](http://www.cs.berkeley.edu/~jordan/papers/paisley-etal-icml12.pdf) J.Paisley, D. Blei, M.Jordan.
  - [Stochastic Variational Inference](http://arxiv.org/pdf/1206.7051.pdf) M. Hoffman, D. Blei, Wang, Paisley. 
  - [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114), Kingma & Welling. 
  - [Stochastic Backpropagation and variational inference in deep latent gaussian models](http://arxiv.org/abs/1401.4082) D. Rezende, S. Mohamed, D. Wierstra.

- **[Lec16](lec16.pdf)** Mar 10: Variational Autoencoders (contd). Normalizing Flows. Adversarial Generative Networks.
  
  *recommended reading*
  - [Semi-supervised learning with Deep generative models](http://arxiv.org/pdf/1406.5298.pdf) Kingma, Rezende, Mohamed, Welling. 
  - [Importance Weighted Autoencoders](http://arxiv.org/abs/1509.00519) Burda, Grosse, Salakhutdinov.
  - [Variational Inference with Normalizing Flows](http://arxiv.org/abs/1505.05770) Rezende, Mohamed. 
  - [Unsupervised Learning using Nonequilibrium Thermodynamics](http://arxiv.org/abs/1503.03585) Sohl-Dickstein et al.
  - [Generative Adversarial Networks](http://arxiv.org/abs/1406.2661), Goodfellow et al. 

- **[Lec17](lec17.pdf)** Mar 29: Adversarial Generative Networks (contd).

  *recommended reading*
  - [Generative Adversarial Networks](http://arxiv.org/abs/1406.2661), Goodfellow et al. 
  - [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](http://arxiv.org/abs/1506.05751) Denton, Chintala, Szlam, Fergus. 
  - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) Radford, Metz, Chintala. 

- **[Lec18](lec18.pdf)** Mar 31: Maximum Entropy Distributions. Self-supervised models (analogies, video prediction, text, word2vec).   

  *recommended reading*
  -  [Graphical Models, Exponential Families and Variational Inference, chapter 3](https://www.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf) M. Wainwright, M. Jordan.
  -  [An Introduction to MCMC for Machine Learning](http://www.cs.ubc.ca/~arnaud/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf) Andrieu, de Freitas, Doucet, Jordan.
  -  [Stochastic relaxation, Gibbs distributions and the Bayesian Restoration of Images](http://www.stat.cmu.edu/~acthomas/724/Geman.pdf) Geman & Geman. 
  -  [Distributed Representations of Words and Phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) Mikolov et al. 
  -  [word2vec Explained: deriving Mikolov et al's negative-sampling embedding method](http://arxiv.org/abs/1402.3722) Goldberg & Levy.

- **Lec19** Apr 5: Non-convex Optimization. Stochastic Optimization. Generalization vs Redundancy. 

- **Lec20** Apr 7:  Guest Lecture (S. Chintala, Facebook AI Research)

- **Lec21** Apr 12: Batch Normalization, Dropout

- **Lec22** Apr 14: Tensor Decompositions, Spin Glasses. Open Problems

- **Lec23** Apr 19: Guest Lecture (Y. Dauphin, Facebook AI Research)

- **Lec24-25**: Oral Presentations
