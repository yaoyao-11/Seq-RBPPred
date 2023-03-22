# Seq-RBPPred
We develop a machine learning model called Seq-RBPPred, mainly using XGBoost[1] train 6944 features. We have two data sets. one is Deep-RBPPred[2],we call it training data set 1, and the other is a merged dataset, which combines data from Deep-RBPPred[2] and the same species in EuRBPDB[3] and PDB[4], we calll it traning data set 2. Seq-RBPPred mainly uses traning data set 2.

![1679154069799](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1679154069799.png)
(A)Flowchart for building training and testing set from EuRBPDB[3] and PDB[4]. Positive samples are from EuRBPDB[3], and negative samples are from PDB[4], confirming the same species in positive and negative samples. According to the sequence of each RBP in the RBPlist, it is extracted from totalFa. If there are multiple corresponding sequences, all are recorded, and the longest one is used. For example, ENSG00000001497 has 5 sequences in totalFa: ENSP00000363940, ENSP00000363944, ENSP00000363937, ENSP00000473471, ENSP00000473630, take the longest sequence ENSP00000363944, and get 115151 RBPs. An initial set of 2777 Non-RBPs was retrieved using PISCES[5] to retain Non-RBPs of the same species as EuRBPDB[40] from the PDB[4], cleave protein sequences and remove those whose function is unknown or associated with RNA binding. Redundancy between the initial 115151 RBPs and 2777 Non-RBPS was removed using the CD-HIT[5] tool with a sequence identity cutoff of 25%, which yielded a non-redundant set of 6,618 RBPs and 1,565 Non-RBPs. To maintain length consistency with proteins in the Non-RBPs set, some proteins were removed from the non-redundant set of RBPs, resulting in a set of 6365 RBPs. Two-thirds are used as the training set and one-third as the testing set, obtaining 4243 RBPs for training, 1043 Non-RBPs for training, and 2122 RBPs and 522 Non-RBPs for testing. After the feature extraction of Protr[6], UniRep[7], SeqVec[8], and ESM-1b[9], some proteins have no result output, so we discarded these proteins, and obtained 3379 RBPs and 1034 Non-RBPs in the training set,1708 RBPs and 517 Non-RBPs in the testing set. Then, we delete the same data in the test set as the Deep-RBPPred[2]
sequence as the final testing set, which contains 1626 RBPs and 329 Non-RBPs. (B)Flowchart for building training set 1 and training set 2. 2780 RBPs and 7093 Non-RBPs are obtained from Deep-RBPPred[2], and then 2,412 RBPs and 6,967 Non-RBPs are obtained after feature extraction by Protr[6], UniRep[7], SeqVec[8], and ESM-1b[9] as our training dataset. Next, the training data set 1 is combined with 3379 RBPs and 1034 Non-RBPs obtained from EuRBPDB[3] and PDB[4], using CD-HIT[5] to remove redundancy and retain sequences with sequence consistency less than or equal to 30%. Finally, 4801 RBPs and 6243 Non-RBPs are obtained as our training data 2.


# Requirements

- Python
- R
- scikit-learn

# Usage

1. Data has our training test and testing test.

# Contact

Contact:[Huang Tao](http://www.sinh.cas.cn/rcdw/qnyjy/202203/t20220310_6387862.html ),Yan Yuyao

# References

1. ```
   1. Chen, T. and C.Guestrin. *Xgboost: A scalable tree boosting system*. in *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*.2016.
   2. Zheng, J., et al., *Deep-RBPPred: Predicting RNA bindingproteins in the proteome scale based on deep learning.* Scientific Reports,2018. **8**(1): p. 15264.
   3. Liao, J.-Y., et al., *EuRBPDB: a comprehensive resource for annotation, functional and oncological investigation of eukaryotic RNA binding proteins (RBPs). * Nucleic Acids Research, 2020. **48**(D1): p. D307-D313.
   4. Berman, H.M., et al., *The Protein Data Bank. * Nucleic Acids Research, 2000. **28**(1): p. 235-242.
   4. Xiao, N., et al., *protr/ProtrWeb: R package and web server for generating various numerical representation schemes of protein sequences. *Bioinformatics, 2015. **31**(11): p.1857-9.
   5. Li, W. and A. Godzik, *Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.* Bioinformatics,2006. **22**(13): p. 1658-9.
   6. Alley, E.C., et al., *Unified rational protein engineering with sequence-based deep representation learning.* Nature Methods, 2019. **16**(12): p. 1315-1322.
   7. Heinzinger, M., et al., *Modeling aspects of the language of life through transfer-learning protein sequences.* BMC Bioinformatics, 2019. **20**(1): p. 723.
   8. Rives, A., et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* Proceedings of the National Academy of Sciences, 2021. **118**(15): p. e2016239118.
   9. Dramiński, M., et al., *Monte Carlo feature selection for supervised classification.* Bioinformatics, 2008. **24**(1): p. 110-117.
   
   
   ```




