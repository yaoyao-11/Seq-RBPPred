import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import math
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

def TrainData(train_file):#'./data/training_set_2.csv'
    data = pd.read_csv(train_file, index_col=0)
    data = data.sample(frac=1, random_state=1)
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    train_x = train.drop(["label"], axis="columns")
    train_y = train["label"]
    test_x = test.drop(["label"], axis="columns")
    test_y = test["label"]

    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    predictor.fit(train_x, train_y)

    return predictor, test_x, test_y

def SVM_prediction(predictor, test_x, test_y):
    result = predictor.predict(test_x)
    accuracy = accuracy_score(test_y, result)  

    newdata_test = {"name": list(test_x.index),
                    "true": list(test_y),
                    "pre": list(result)}
    newdata_test = pd.DataFrame(newdata_test)
    for i in newdata_test.index:
        if newdata_test.loc[i, "pre"] == newdata_test.loc[i, "true"]:
            newdata_test.loc[i, "prediction"] = "T"
        else:
            newdata_test.loc[i, "prediction"] = "F"
    for i in newdata_test.index:
        if newdata_test.loc[i, "true"] == 1:
            newdata_test.loc[i, "RBP"] = "P"
        else:
            newdata_test.loc[i, "RBP"] = "N"
    newdata_test["ACC_MCC"] = "NULL"
    for i in newdata_test.index:
        if newdata_test.iloc[i, -2] == "P":
            if newdata_test.iloc[i, -3] == "T":
                newdata_test.iloc[i, -1] = "TP"
            else:
                newdata_test.iloc[i, -1] = "FN"
        if newdata_test.iloc[i, -2] == "N":
            if newdata_test.iloc[i, -3] == "T":
                newdata_test.iloc[i, -1] = "TN"
            else:
                newdata_test.iloc[i, -1] = "FP"
    prediction = newdata_test["prediction"].values.tolist()
    T_F = dict(zip(*np.unique(prediction, return_counts=True)))
    count = T_F["T"] / len(prediction)
    ACC_MCC_list = newdata_test["ACC_MCC"].values.tolist()
    ACC_MCC = dict(zip(*np.unique(ACC_MCC_list, return_counts=True)))
    TP = ACC_MCC["TP"]
    TN = ACC_MCC["TN"]
    FP = ACC_MCC["FP"]
    FN = ACC_MCC["FN"]
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    return ACC, MCC, SN, SP, accuracy

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', font=Path('TNR.ttf'), fontsize=25)
    plt.ylabel('True Positive Rate', font=Path('TNR.ttf'), fontsize=25)
    plt.title('Receiver operating characteristic example', font=Path('TNR.ttf'), fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    plt.show()

def main():
    train_file = './data/training_set_2.csv'
    predictor, test_x, test_y = TrainData(train_file)

    print("Training Metrics:")
    ACC_train, MCC_train, SN_train, SP_train, accuracy_train = SVM_prediction(predictor, test_x, test_y)
    print("ACC:", ACC_train)
    print("MCC:", MCC_train)
    print("SN:", SN_train)
    print("SP:", SP_train)
    print("Accuracy:", accuracy_train)

    # Predict on Independent Test
    test_file = "./data/independent_test.csv"
    ACC_test, MCC_test, SN_test, SP_test, accuracy_test = SVM_prediction(predictor, test_x, test_y)

    print("\nTest Metrics:")
    print("ACC:", ACC_test)
    print("MCC:", MCC_test)
    print("SN:", SN_test)
    print("SP:", SP_test)
    print("Accuracy:", accuracy_test)

    # Plot ROC curve for Independent Test
    data_test = pd.read_csv(test_file, index_col=0)
    test_independent_x = data_test.drop(["label"], axis="columns")
    test_independent_y = data_test["label"].astype(int)
    result = predictor.decision_function(test_independent_x)
    acu_curve(test_independent_y, result)

if __name__ == "__main__":
    main()
