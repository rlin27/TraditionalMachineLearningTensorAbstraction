# Tensor Abstraction of Traditional ML Models - Decision Tree / SVM

## Decision Tree
1. Decision Tree Induction in High Dimensional, Hierarchically Distributed Databases (SIAM 2005) [[PDF](https://epubs.siam.org/doi/10.1137/1.9781611972757.42)]
2. Scikit-learn: Machine Learning in Python (JMLR 2011) [[PDF](https://arxiv.org/pdf/1201.0490.pdf)]
3. Parallel boosted regression trees for web search ranking (W3 2011) [[PDF](https://dl.acm.org/doi/10.1145/1963405.1963461)]
4. XGBoost : Reliable Large-scale Tree Boosting System (SIGKDD 2015) [[PDF](https://www.semanticscholar.org/paper/XGBoost-%3A-Reliable-Large-scale-Tree-Boosting-System-Chen-Guestrin/f0d90cfd564a2ec6281ad58b58aef838decb2fe4)]
5. XGBoost: A Scalable Tree Boosting System (SIGKDD 2016) [[PDF](https://www.semanticscholar.org/paper/XGBoost%3A-A-Scalable-Tree-Boosting-System-Chen-Guestrin/26bc9195c6343e4d7f434dd65b4ad67efe2be27a)]
6. LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NeurIPS 2017) [[PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)]
7. GPU-acceleration for Large-scale Tree Boosting (ArXiv 2017) [[PDF](https://arxiv.org/pdf/1706.08359.pdf)]
8. XBART: Accelerated Bayesian Additive Regression Trees (PMLR 2019) [[PDF](https://arxiv.org/pdf/1810.02215.pdf)]
9. Compiling Classical ML Pipelines into Tensor Computations for One-size-fits-all Prediction Serving (NeurIPs Workshop 2019) [[PDF](http://learningsys.org/neurips19/assets/papers/27_CameraReadySubmission_Hummingbird%20(5).pdf)][[GitHub](https://github.com/microsoft/hummingbird)]
> - This paper introduces three strategies for decision tree tensorization, namely, GEMM, TreeTraversal, and PerfectTreeTraversal.
10. A Tensor Compiler for Unified Machine Learning Prediction Serving (OSDI 2020) [[PDF](https://www.usenix.org/system/files/osdi20-nakandala.pdf)][[GitHub](https://github.com/microsoft/hummingbird)]
> - This paper is an enriched version of the 9th paper.
> - The main issue in HummingBird is that it is heuristic to decide which strategy will be employed when given a tree. It will be appreciate if there can be some theorems to help users decide the strategy automatically.

## Support Vector Machine (SVM)
1. Supervised tensor learning (ICDM 2005) [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1565711)]
> - This paper generalizes support vector machines, minimax probalility machine, Fisher discriminant analysis, and distance metric learning to support tensor machines, tensor minimax probabilities, and the multiple distance metrics learning, respectively.
2. Support tucker machines (CVPR 2011) [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5995663)]
> - This paper addresses the two-class classification problem within the tensor-based framework, by formulating the Support Tucker Machines (STuM).
3. A Linear Support Higher-Order Tensor Machine for Classification (IEEE Transactions on Image Processing 2013) [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6482624&tag=1)]
> - This paper presents a novel linear support higher-order tensor machine (SHTM) which integrates the mertis of linear C-support vector machine (C-SVM) and tensor rank-one decomposition.
4. A support tensor train machine (IJCNN 2019) [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8851985)]
> - The previous work STM lacks the expressive power due to its rank-one tensor constraint, and STuM is not scalable because of the exponentially sized Tucker core tensor. To overcome thesis limitations, a novel and effective support tensor train machine (STTM) is proposed by employing a general and scalable tensor train as the parameter model.
5. Exploiting tensor networks for efficient machine learning (PhD Thesis 2021) [[PDF](https://hub.hku.hk/handle/10722/308618)]
> - This thesis explores the tensorization and compression of machine learning models (SVMs and RBMs).
6. Kernelized Support Tensor Train Machines (Pattern Recognition 2022) [[PDF](https://arxiv.org/pdf/2001.00360.pdf)]
> - The proposed K-STTM allows STTM to work with kernel tricks.

**However**, the motivation of the papers listed above is to deal with real-life high-order data (e.g., RGB images, videos), rather than tensorize the traditional machine learning models themselves to accelerate their execution. Moreover, their codes are based on MATLAB, which are not designed for GPUs. In conclusion, the tensorization mentioned in those papers is not quite consistent with the tensorization in the Tensor-native-Data-Science. But it is still possible for us to learn something useful from the idea in those papers.

## Existing Repos
1. [MATLAB Tree](https://www.mathworks.com/help/stats/decision-trees.html) & [MATLAB](https://www.mathworks.com/discovery/support-vector-machine.html)
2. [Sklearn](https://scikit-learn.org/stable/)
3. [cuML](https://docs.rapids.ai/api/cuml/stable/)
4. [HummingBird](https://github.com/microsoft/hummingbird)

> - MATLAB and Sklearn are implementation based on CPUs, the models defined by them lack the abilibty to be deployed on GPUs. 
> - While cuML and HummingBird both aim to tensorize traiditional machine leraning models defined by Sklearn, therefore, allow the models to run on GPUs.

