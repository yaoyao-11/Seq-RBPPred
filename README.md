## **Seq-RBPPred**

RNA-binding proteins (RBPs) can interact with RNAs to regulate RNA translation, modification, splicing, and other important biological processes. The accurate identification of RBPs is of paramount importance for gaining insights into the intricate mechanisms underlying organismal life activities. Traditional experimental methods to predict RBPs need to invest a lot of time and money, so it is important to develop computational methods to predict RBPs. However, the existing approaches for RBP prediction still require further improvement due to the unidentified RBPs in many species. In this study, we present Seq-RBPPred (Predicting RNA-binding proteins from Sequences), a novel method that utilizes a comprehensive feature representation encompassing both biophysical properties and hidden-state features derived from protein sequences. In the results, comprehensive performance evaluations of Seq-RBPPred its superiority compare with state-of-the-art methods.
![1700815233083](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1700815233083.png)

We develop a machine learning model called Seq-RBPPred, mainly using XGBoost[1] train 6944 features. We have two data sets. one is Deep-RBPPred[2],we call it training data set 1, and the other is a merged dataset, which combines data from Deep-RBPPred and the same species in EuRBPDB[3] and PDB[4], we call it traning data set 2. Seq-RBPPred mainly uses traning data set 2.

![1700815121703](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\1700815121703.png)


Build training and testing sets: (A)Flowchart for building training and testing set from EuRBPDB and PDB. Positive samples are from EuRBPDB, and negative samples are from PDB, confirming the same species in positive and negative samples. According to the sequence of each RBP in the RBPlist, it is extracted from totalFa. If there are multiple corresponding sequences, all are recorded, and the longest one is used. For example, ENSG00000001497 has 5 sequences in totalFa: ENSP00000363940, ENSP00000363944, ENSP00000363937, ENSP00000473471, ENSP00000473630, take the longest sequence ENSP00000363944, and get 115151 RBPs. An initial set of 2777 Non-RBPs was retrieved using PISCES[5] to retain Non-RBPs of the same species as EuRBPDB from the PDB, cleave protein sequences and remove those whose function is unknown or associated with RNA binding. Redundancy between the initial 115151 RBPs and 2777 Non-RBPS was removed using the CD-HIT[6] tool with a sequence identity cutoff of 25%, which yielded a non-redundant set of 6,618 RBPs and 1,565 Non-RBPs. To maintain length consistency with proteins in the Non-RBPs set, some proteins were removed from the non-redundant set of RBPs, resulting in a set of 6365 RBPs. Two-thirds are used as the training set and one-third as the testing set, obtaining 4243 RBPs for training, 1043 Non-RBPs for training, and 2122 RBPs and 522 Non-RBPs for testing. After the feature extraction of Protr[7], UniRep[8], SeqVec[9], and ESM-1b[10], some proteins have no result output, so we discarded these proteins, and obtained 3379 RBPs and 1034 Non-RBPs in the training set,1708 RBPs and 517 Non-RBPs in the testing set. Then, we delete the same data in the test set as the Deep-RBPPred sequence as the final testing set, which contains 1626 RBPs and 329 Non-RBPs. (B)Flowchart for building training set 1 and training set 2. 2780 RBPs and 7093 Non-RBPs are obtained from Deep-RBPPred, and then 2,412 RBPs and 6,967 Non-RBPs are obtained after feature extraction by Protr, UniRep, SeqVec, and ESM-1b as our training dataset. Next, the training data set 1 is combined with 3379 RBPs and 1034 Non-RBPs obtained from EuRBPDB and PDB, using CD-HIT to remove redundancy and retain sequences with sequence consistency less than or equal to 30%. Finally, 4801 RBPs and 6243 Non-RBPs are obtained as our training data 2.


# Requirements

- Python
- R
- scikit-learn

# Usage

1. Data has our training test and testing test.

2. You can use Seq-RBPPred to predict RBPs.

3. Train model: run ./Seq-RBPPred/XGBoost_train.ipynb

   You can get the  ./Seq-RBPPred/model.pickle.dat

4. model=pickle.load("./Seq-RBPPred/model.pickle.dat")

   y_test_pred = model.predict(test_x)

   test_x is your data, you get test_x by using ./Get Features to get 6944 features for each protein.

# Contact

Contact: YuYao Yan(18853818507@163.com)

# References

1. Chen, T. and C.Guestrin. *Xgboost: A scalable tree boosting system*. in *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*.2016.
2. Zheng, J., et al., *Deep-RBPPred: Predicting RNA bindingproteins in the proteome scale based on deep learning.* Scientific Reports,2018. **8**(1): p. 15264.
3. Liao, J.-Y., et al., EuRBPDB: a comprehensive resource for annotation, functional and oncological investigation of eukaryotic RNA binding proteins (RBPs). Nucleic Acids Research, 2020. **48**(D1): p. D307-D313.
4. Berman, H.M., et al., The Protein Data Bank. Nucleic Acids Research, 2000. **28**(1): p. 235-242.
5. Wang, G. and R.L. Dunbrack, Jr., PISCES: a protein sequence culling server. Bioinformatics, 2003. 19(12): p. 1589-91.
6. Li, W. and A. Godzik, *Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.* Bioinformatics,2006. **22**(13): p. 1658-9.
7. Xiao, N., et al., protr/ProtrWeb: R package and web server for generating various numerical representation schemes of protein sequences. Bioinformatics, 2015. **31**(11): p.1857-9.
8. Alley, E.C., et al., *Unified rational protein engineering with sequence-based deep representation learning.* Nature Methods, 2019. **16**(12): p. 1315-1322.
9. Heinzinger, M., et al., *Modeling aspects of the language of life through transfer-learning protein sequences.* BMC Bioinformatics, 2019. **20**(1): p. 723.
10. Rives, A., et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* Proceedings of the National Academy of Sciences, 2021. **118**(15): p. e2016239118.

