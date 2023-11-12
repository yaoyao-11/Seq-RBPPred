## Data File Descriptions

1. Deep_RBPpred

   This folder contains data on RNA-binding proteins (RBPs) and non-RBPs from  DeepRBPPred[1]. This corresponds to Dataset 1 in our paper.

2. EuRBPDB_PDB_data

   This includes RBP and non-RBP data sourced from the EuRBPDB[2] and PDB[3] databases. When combined with the data from the Deep_RBPpred folder, it forms Dataset 2 in our paper.

3. How to Use the Data

   - DeepRBPPred

     The methods from the DeepRBPPred can be employed to make predictions on the independent_test data.

   - Deep_forest，DmLab,Random_forest,SVM,Seq-RBPPred

     Datasets 1 and 2 can be utilized for training to create two distinct models. These models can then be applied to make predictions on the independent_test data. The corresponding codes for each method are located in the respective folders: Deep_forest, DmLab, Random_forest, SVM, Seq-RBPPred.

     #### 

#### References

1. Zheng, J., et al., *Deep-RBPPred: Predicting RNA bindingproteins in the proteome scale based on deep learning.* Scientific Reports,2018. **8**(1): p. 15264.
2. Liao, J.-Y., et al., EuRBPDB: a comprehensive resource for annotation, functional and oncological investigation of eukaryotic RNA binding proteins (RBPs).  Nucleic Acids Research, 2020. **48**(D1): p. D307-D313.
3. Berman, H.M., et al., The Protein Data Bank.  Nucleic Acids Research, 2000. **28**(1): p. 235-242.





