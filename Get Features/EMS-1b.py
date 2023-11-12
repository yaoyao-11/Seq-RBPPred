#ESM-1b: Predicting structure, function, and other protein properties directly from a single sequence
#Only keep sequence length < 1023

from transformers import ESMForMaskedLM, ESMTokenizer
import pandas as pd
tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False )
model = ESMForMaskedLM.from_pretrained("facebook/esm-1b")

file="yaoyao-11/Seq-RBPpred/data/test_NonRBPs_522.fasta"
# file="yaoyao-11/Seq-RBPpred/data/test_RBPs_2122.fasta"
# file="yaoyao-11/Seq-RBPpred/data/train_RBPs_4243.fasta"
# file="yaoyao-11/Seq-RBPpred/data/train_NonRBPs_1043.fasta"

names=[]
sequences=[]
fa_dict={}
with open(file) as fa:
    for line in fa:
        line=line.replace("\n", "")
        if line.startswith(">"):
            #delete ">"
            seq_name=line[1:].split(" ")[0]
            fa_dict[seq_name]=""
            names.append(seq_name)
        else:
            fa_dict[seq_name] += line.replace('\n','')
            sequences.append(fa_dict[seq_name])

fix_length_sequences=[]
last_names=[]
for i in fa_dict.keys():
    if len(fa_dict[i])<1023:
        last_names.append(i)
        fix_length_sequences.append(fa_dict[i])


last_features=[]
for i in tqdm(fix_length_sequences):
    encoded_input = tokenizer(i, return_tensors='pt')
    output = model(**encoded_input)
    data=output[0].mean(axis=1)
    last_features.append(data[0].detach().numpy().tolist())

#encoded_input = tokenizer(sequences, return_tensors='pt',padding=True,truncation=True)
#output = model(**encoded_input)
#data=output[0].mean(axis=1)
#last_features=[]
#for i in range(len(data)):
 #   last_features.append(data[i].detach().numpy().tolist())

columns_name=[]
for i in range(len(last_features[0])):
    columns_name.append("features_{}".format(i))
need_data=pd.DataFrame(columns=columns_name,index=last_names,data=last_features)
need_data.to_csv('yaoyao-11/Seq-RBPpred/ESM-1b/test_NonRBPs_522.csv',encoding='gbk')
