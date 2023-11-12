## Protr: Obtain a series of physical and chemical properties
## language:R
## Take yaoyao-11/Seq-RBPpred/train RBPs_4243.fasta as an example

library(protr)
library(magrittr)
# setwd("./")
setwd="./data/"
N_file="Deep_RBPpred/nonRBP7093.fasta"
# N_file="train Non-RBPs_1043.fasta"


x <- readFASTA(N_file)
m <- matrix(unlist(x),byrow=TRUE,ncol=length(x[[1]]))
rownames(m) <- names(x)
data=as.data.frame(m)
#rownames(data)
data$name <- rownames(data)
rownames(data) <- NULL
## extractAAC
#make data frame
a <- extractAAC(data[100,1]) 
a <- t(as.data.frame(a))
c=list()

X_c <- ""
for(i in 1:nrow(data)){
  X_s <- unlist(strsplit(data[i,1],""))
  if("X" %in% X_s){
    print(i)
    c=c(c,data[i,'name'])
    X_c <- c(X_c,i)
  }
  
}

U_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("U" %in% U_s){
    print(i)
    c=c(c,data[i,'name'])
    U_c <- c(U_c,i)
  }
  
}

specific_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("*" %in% U_s){
    print(i)
    c=c(c,data[i,'name'])
    specific_c <- c(specific_c,i)
  }
  
}

O_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("O" %in% U_s){
    print(i)
    c=c(c,data[i,'name'])
    O_c <- c(O_c,i)
  }
  
}

U_c <-  U_c[-1] %>% as.numeric()
X_c <- X_c[-1] %>% as.numeric()
specific_c <- specific_c[-1] %>% as.numeric()
O_c<- O_c[-1] %>% as.numeric()

data_no_X <- data[-c(X_c,U_c,specific_c,O_c),] 
data_no_X_d=data_no_X

aaOut.o <- a
for (i in 1:nrow(data_no_X_d)) {
    aaOut <- extractAAC(data_no_X_d[i,1]) %>% data.frame() %>%t
    rownames(aaOut) <-data_no_X_d[i,2] 
    aaOut.o<- rbind(aaOut.o,aaOut)
    print(i)
}

aaOut.out=aaOut.o[-1,]
write.csv(aaOut.out,"/train_N_AAC.csv")
# write.csv(aaOut.out,"./train_N_AAC.csv")

## extractPAAC
myprops <- data.frame(
  AccNo = c("MyProp1", "MyProp2", "MyProp3"),
  A = c(0.62, -0.5, 15), R = c(-2.53, 3, 101),
  N = c(-0.78, 0.2, 58), D = c(-0.9, 3, 59),
  C = c(0.29, -1, 47), E = c(-0.74, 3, 73),
  Q = c(-0.85, 0.2, 72), G = c(0.48, 0, 1),
  H = c(-0.4, -0.5, 82), I = c(1.38, -1.8, 57),
  L = c(1.06, -1.8, 57), K = c(-1.5, 3, 73),
  M = c(0.64, -1.3, 75), F = c(1.19, -2.5, 91),
  P = c(0.12, 0, 42), S = c(-0.18, 0.3, 31),
  T = c(-0.05, -0.4, 45), W = c(0.81, -3.4, 130),
  Y = c(0.26, -2.3, 107), V = c(1.08, -1.5, 43)
)

# if(X_s[i] == "X"){
#   print(i)
#   X_c <- c(X_c,i)
# }


X_c <- ""
for(i in 1:nrow(data)){
  X_s <- unlist(strsplit(data[i,1],""))
  if("X" %in% X_s){
    print(i)
    X_c <- c(X_c,i)
  }
  
}

U_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("U" %in% U_s){
    print(i)
    U_c <- c(U_c,i)
  }
  
}
#去掉*???
specific_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("*" %in% U_s){
    print(i)
    specific_c <- c(specific_c,i)
  }
  
}

#去掉O
O_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("O" %in% U_s){
    print(i)
    O_c <- c(O_c,i)
  }
  
}

U_c <-  U_c[-1] %>% as.numeric()

X_c <- X_c[-1] %>% as.numeric()
specific_c <- specific_c[-1] %>% as.numeric()
O_c<- O_c[-1] %>% as.numeric()

data_no_X <- data[-c(X_c,U_c,specific_c,O_c),] 

#unlist(strsplit(data[5,1],""))[1]


#Error in extractPAAC(data_no_X[i, 1], lambda = 1, customprops = myprops,  : x has unrecognized amino acid type
#number:15109
data_no_X_d <- data_no_X
# data_no_X_d=data

#make data frame
a <- extractPAAC(data[10,1],lambda = 1,customprops = myprops, props = c("Hydrophobicity", "Hydrophilicity", "SideChainMass", "CIDH920105", "BHAR880101","CHAM820101", "CHAM820102","MyProp1", "MyProp2", "MyProp3"))
a <- t(as.data.frame(a))

aaOut.o <- a
for (i in 1:nrow(data_no_X_d)) {
  if(length(unlist(strsplit(data_no_X_d[i,1],""))) >1){
    aaOut <- extractPAAC(data_no_X_d[i,1],lambda = 1,customprops = myprops,props = c("Hydrophobicity", "Hydrophilicity", "SideChainMass",
                                                                                         "CIDH920105", "BHAR880101","CHAM820101", "CHAM820102",
                                                                                    "MyProp1", "MyProp2", "MyProp3")) %>% data.frame() %>%t
    rownames(aaOut) <-data_no_X_d[i,2] 
    aaOut.o<- rbind(aaOut.o,aaOut)
  }
  print(i)
#  aaOut[i] <- rownames(data)[i]  
}

aaOut.out=aaOut.o[-1,]
write.csv(aaOut.out,"./train_N_PAAC.csv")
# write.csv(aaOut.out,"./train_N_PAAC.csv")

## extractCTDC
#make data frame
a <- extractCTDC(data[10,1]) 
a <- t(as.data.frame(a))
X_c <- ""
for(i in 1:nrow(data)){
  X_s <- unlist(strsplit(data[i,1],""))
  if("X" %in% X_s){
    print(i)
    X_c <- c(X_c,i)
  }
}

U_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("U" %in% U_s){
    print(i)
    U_c <- c(U_c,i)
  }
}

specific_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("*" %in% U_s){
    print(i)
    specific_c <- c(specific_c,i)
  }
}

#去掉O
O_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("O" %in% U_s){
    print(i)
    O_c <- c(O_c,i)
  }
}

U_c <-  U_c[-1] %>% as.numeric()
X_c <- X_c[-1] %>% as.numeric()
specific_c <- specific_c[-1] %>% as.numeric()
O_c<- O_c[-1] %>% as.numeric()

data_no_X <- data[-c(X_c,U_c,specific_c,O_c),]  
data_no_X_d=data_no_X
# data_no_X_d=data

aaOut.o <- a
for (i in 1:nrow(data_no_X_d)) {
  aaOut <- extractCTDC(data_no_X_d[i,1]) %>% data.frame() %>%t
  rownames(aaOut) <-data_no_X_d[i,2] 
  aaOut.o<- rbind(aaOut.o,aaOut)
  print(i)
  #  aaOut[i] <- rownames(data)[i]  
}

aaOut.out=aaOut.o[-1,]
write.csv(aaOut.out,"./train_N_CTDC.csv")
# write.csv(aaOut.out,"./train_N_CTDC.csv")

## extractCTDT
a <- extractCTDT(data[10,1]) 
a <- t(as.data.frame(a))
X_c <- ""
for(i in 1:nrow(data)){
  X_s <- unlist(strsplit(data[i,1],""))
  if("X" %in% X_s){
    print(i)
    X_c <- c(X_c,i)
  }
}

U_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("U" %in% U_s){
    print(i)
    U_c <- c(U_c,i)
  }
}

specific_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("*" %in% U_s){
    print(i)
    specific_c <- c(specific_c,i)
  }
}

#去掉O
O_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("O" %in% U_s){
    print(i)
    O_c <- c(O_c,i)
  }
}


U_c <-  U_c[-1] %>% as.numeric()
X_c <- X_c[-1] %>% as.numeric()
specific_c <- specific_c[-1] %>% as.numeric()
O_c<- O_c[-1] %>% as.numeric()
data_no_X <- data[-c(X_c,U_c,specific_c,O_c),] 
data_no_X_d=data_no_X
# data_no_X_d=data

aaOut.o <- a
for (i in 1:nrow(data_no_X_d)) {
  aaOut <- extractCTDT(data_no_X_d[i,1]) %>% data.frame() %>%t
  rownames(aaOut) <-data_no_X_d[i,2] 
  aaOut.o<- rbind(aaOut.o,aaOut)
  print(i)
  #  aaOut[i] <- rownames(data)[i]  
}

aaOut.out=aaOut.o[-1,]
write.csv(aaOut.out,"./train_N_CTDT.csv")
# write.csv(aaOut.out,"./train_N_CTDT.csv")

## extractCTDD
a <- extractCTDD(data[10,1]) 
a <- t(as.data.frame(a))
X_c <- ""
for(i in 1:nrow(data)){
  X_s <- unlist(strsplit(data[i,1],""))
  if("X" %in% X_s){
    print(i)
    X_c <- c(X_c,i)
  }
}

U_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("U" %in% U_s){
    print(i)
    U_c <- c(U_c,i)
  }
}

specific_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("*" %in% U_s){
    print(i)
    specific_c <- c(specific_c,i)
  }
}
#去掉O
O_c <- ""
for(i in 1:nrow(data)){
  U_s <- unlist(strsplit(data[i,1],""))
  if("O" %in% U_s){
    print(i)
    O_c <- c(O_c,i)
  }
}

U_c <-  U_c[-1] %>% as.numeric()
X_c <- X_c[-1] %>% as.numeric()
specific_c <- specific_c[-1] %>% as.numeric()
O_c<- O_c[-1] %>% as.numeric()
data_no_X <- data[-c(X_c,U_c,specific_c,O_c),] 
data_no_X_d=data_no_X
# data_no_X_d=data

aaOut.o <- a
for (i in 1:nrow(data_no_X_d)) {
  aaOut <- extractCTDD(data_no_X_d[i,1]) %>% data.frame() %>%t
  rownames(aaOut) <-data_no_X_d[i,2] 
  aaOut.o<- rbind(aaOut.o,aaOut)
  print(i)
  #  aaOut[i] <- rownames(data)[i]  
}

aaOut.out=aaOut.o[-1,]
write.csv(aaOut.out,"./train_N_CTDD.csv")
# write.csv(aaOut.out,"./train_N_CTDD.csv")
