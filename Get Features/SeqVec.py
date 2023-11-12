# SeqVec: Effectively captured the biophysical properties
# Take yaoyao-11/Seq-RBPpred/train_RBPs_4243.fasta as an example
import numpy as np
import pandas as pd

data = np.load("yaoyao-11/Seq-RBPpred/data/train_RBPs_4243_embeddings.npz") 
features_names=[]
features=[]
for test in list(data.items()):
    name=test[0].split(" ")[0]
    features_names.append(name)
    features.append(test[1].tolist())
columns_name=[]
for i in range(len(features[0])):
    columns_name.append("features_{}".format(i))
need_data=pd.DataFrame(columns=columns_name,index=features_names,data=features)
need_data.to_csv('D:/seqvec/result/train_RBPs_4243_features.csv',encoding='gbk')