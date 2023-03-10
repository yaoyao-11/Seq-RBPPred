# bind train features
setwd("yaoyao-11/Seq-RBPpred/Protr_file/train_feature")
P_ACC=read.csv("./train_P_AAC.csv",header=T,row.names=1)
N_ACC=read.csv("./train_N_AAC.csv",header=T,row.names=1)
P_PACC=read.csv("./train_P_PAAC.csv",header=T,row.names=1)
N_PACC=read.csv("./train_N_PAAC.csv",header=T,row.names=1)
P_CTDC=read.csv("./train_P_CTDC.csv",header=T,row.names=1)
N_CTDC=read.csv("./train_N_CTDC.csv",header=T,row.names=1)
P_CTDT=read.csv("./train_P_CTDT.csv",header=T,row.names=1)
N_CTDT=read.csv("./train_N_CTDT.csv",header=T,row.names=1)
P_CTDD=read.csv("./train_P_CTDD.csv",header=T,row.names=1)
N_CTDD=read.csv("./train_N_CTDD.csv",header=T,row.names=1)

t_P_ACC=t(P_ACC)
t_N_ACC=t(N_ACC)
t_P_PACC=t(P_PACC)
t_P_PACC=t_P_PACC[-21,]
t_N_PACC=t(N_PACC)
t_N_PACC=t_N_PACC[-21,]
t_P_CTDC=t(P_CTDC)
t_N_CTDC=t(N_CTDC)
t_P_CTDT=t(P_CTDT)
t_N_CTDT=t(N_CTDT)
t_P_CTDD=t(P_CTDD)
t_N_CTDD=t(N_CTDD)

ACC=cbind(t_P_ACC, t_N_ACC)
ACC=data.frame(ACC,stringsAsFactors = F,check.names = F) 
PACC=cbind(t_P_PACC,t_N_PACC)
PACC=data.frame(PACC,stringsAsFactors = F,check.names = F) 
PACC_ACC=dplyr::bind_rows(ACC, PACC)
CTDC=cbind(t_P_CTDC, t_N_CTDC)
CTDC=data.frame(CTDC,stringsAsFactors = F,check.names = F) 
CTDT=cbind(t_P_CTDT, t_N_CTDT)
CTDT=data.frame(CTDT,stringsAsFactors = F,check.names = F) 
CTDD=cbind(t_P_CTDD, t_N_CTDD)
CTDD=data.frame(CTDD,stringsAsFactors = F,check.names = F) 
CTD=dplyr::bind_rows(CTDC, CTDT,CTDD,)

ALL=dplyr::bind_rows(PACC_ACC,CTD)

write.csv(ALL,"./ALL_protr_train.CSV",quote = FALSE)

# bind test features
setwd("yaoyao-11/Seq-RBPpred/Protr_file/test_feature")
P_ACC=read.csv("./test_P_AAC.csv",header=T,row.names=1)
N_ACC=read.csv("./test_N_AAC.csv",header=T,row.names=1)
P_PACC=read.csv("./test_P_PAAC.csv",header=T,row.names=1)
N_PACC=read.csv("./test_N_PAAC.csv",header=T,row.names=1)
P_CTDC=read.csv("./test_P_CTDC.csv",header=T,row.names=1)
N_CTDC=read.csv("./test_N_CTDC.csv",header=T,row.names=1)
P_CTDT=read.csv("./test_P_CTDT.csv",header=T,row.names=1)
N_CTDT=read.csv("./test_N_CTDT.csv",header=T,row.names=1)
P_CTDD=read.csv("./test_P_CTDD.csv",header=T,row.names=1)
N_CTDD=read.csv("./test_N_CTDD.csv",header=T,row.names=1)

t_P_ACC=t(P_ACC)
t_N_ACC=t(N_ACC)
t_P_PACC=t(P_PACC)
t_P_PACC=t_P_PACC[-21,]
t_N_PACC=t(N_PACC)
t_N_PACC=t_N_PACC[-21,]
t_P_CTDC=t(P_CTDC)
t_N_CTDC=t(N_CTDC)
t_P_CTDT=t(P_CTDT)
t_N_CTDT=t(N_CTDT)
t_P_CTDD=t(P_CTDD)
t_N_CTDD=t(N_CTDD)

ACC=cbind(t_P_ACC, t_N_ACC)
ACC=data.frame(ACC,stringsAsFactors = F,check.names = F) 
PACC=cbind(t_P_PACC,t_N_PACC)
PACC=data.frame(PACC,stringsAsFactors = F,check.names = F) 
PACC_ACC=dplyr::bind_rows(ACC, PACC)
CTDC=cbind(t_P_CTDC, t_N_CTDC)
CTDC=data.frame(CTDC,stringsAsFactors = F,check.names = F) 
CTDT=cbind(t_P_CTDT, t_N_CTDT)
CTDT=data.frame(CTDT,stringsAsFactors = F,check.names = F) 
CTDD=cbind(t_P_CTDD, t_N_CTDD)
CTDD=data.frame(CTDD,stringsAsFactors = F,check.names = F) 
CTD=dplyr::bind_rows(CTDC, CTDT,CTDD,)

ALL=dplyr::bind_rows(PACC_ACC,CTD)

write.csv(ALL,"./ALL_protr_test.CSV",quote = FALSE)

