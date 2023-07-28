# AS-GCN
Paper: Classifying multiclass relationships between ASes using graph convolutional network

Paper Link: https://link.springer.com/article/10.1007/s42524-022-0217-1



## Abstract

Precisely understanding the business relationships between autonomous systems (ASes) is essential for studying the Internet structure. To date, many inference algorithms, which mainly focus on peer-to-peer (P2P) and provider-to-customer (P2C) binary classification, have been proposed to classify the AS relationships and have achieved excellent results. However, business-based sibling relationships and structure-based exchange relationships have become an increasingly nonnegligible part of the Internet market in recent years. Existing algorithms are often difficult to infer due to the high similarity of these relationships to P2P or P2C relationships. In this study, we focus on multiclassification of AS relationship for the first time. We first summarize the differences between AS relationships under the structural and attribute features, and the reasons why multiclass relationships are difficult to be inferred. We then introduce new features and propose a graph convolutional network (GCN) framework, AS-GCN, to solve this multiclassification problem under complex scenes. The proposed framework considers the global network structure and local link features concurrently. Experiments on real Internet topological data validate the effectiveness of our method, that is, AS-GCN. The proposed method achieves comparable results on the binary classification task and outperforms a series of baselines on the more difficult multiclassification task, with an overall metrics above 95%.



## Background

<img src="C:\Users\72436\AppData\Roaming\Typora\typora-user-images\image-20230728135006635.png" alt="image-20230728135006635" style="zoom: 80%;" />



## Model

![image-20230728135056446](C:\Users\72436\AppData\Roaming\Typora\typora-user-images\image-20230728135056446.png)



```
@article{peng2022classifying,
  title={Classifying multiclass relationships between ASes using graph convolutional network},
  author={Peng, Songtao and Shu, Xincheng and Ruan, Zhongyuan and Huang, Zegang and Xuan, Qi},
  journal={Frontiers of Engineering Management},
  volume={9},
  number={4},
  pages={653--667},
  year={2022},
  publisher={Springer}
}
```



