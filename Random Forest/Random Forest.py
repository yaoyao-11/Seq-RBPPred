import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from deepforest import CascadeForestClassifier
from pathlib import Path
from matplotlib import rcParams

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
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', font=Path('TNR.ttf'), fontsize=25)
    plt.ylabel('True Positive Rate', font=Path('TNR.ttf'), fontsize=25)
    plt.title('Receiver operating characteristic example', font=Path('TNR.ttf'), fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    plt.show()

def TrainData(train_file):
    # data = pd.read_csv('./data/training_set_2.csv', index_col=0)
    data = pd.read_csv(train_file, index_col=0)
    data = data.sample(frac=1, random_state=1)
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    train_x = train.drop(["label"], axis="columns")
    train_y = train["label"]
    test_x = test.drop(["label"], axis="columns")
    test_y = test["label"]

    forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)
    forest.fit(train_x, train_y)
    y_test_pred = forest.predict(test_x)
    y_test_pred = list(y_test_pred)

    accuracy_score(test_y, y_test_pred)
    newdata = {"name": list(test_x.index),
               "true": list(test_y),
               "pre": list(y_test_pred)}
    newdata = pd.DataFrame(newdata)
    for i in newdata.index:
        if newdata.loc[i, "pre"] == newdata.loc[i, "true"]:
            newdata.loc[i, "prediction"] = "T"
        else:
            newdata.loc[i, "prediction"] = "F"
    for i in newdata.index:
        if newdata.loc[i, "pre"] == newdata.loc[i, "true"]:
            newdata.loc[i, "prediction"] = "T"
        else:
            newdata.loc[i, "prediction"] = "F"
    for i in newdata.index:
        if newdata.loc[i, "true"] == 1:
            newdata.loc[i, "RBP"] = "P"
        else:
            newdata.loc[i, "RBP"] = "N"
    newdata["ACC_MCC"] = "NULL"
    for i in newdata.index:
        if newdata.iloc[i, -2] == "P":
            if newdata.iloc[i, -3] == "T":
                newdata.iloc[i, -1] = "TP"
            else:
                newdata.iloc[i, -1] = "FN"
        if newdata.iloc[i, -2] == "N":
            if newdata.iloc[i, -3] == "T":
                newdata.iloc[i, -1] = "TN"
            else:
                newdata.iloc[i, -1] = "FP"
    prediction = newdata["prediction"].values.tolist()
    T_F = dict(zip(*np.unique(prediction, return_counts=True)))
    count = T_F["T"] / len(prediction)
    ACC_MCC_list = newdata["ACC_MCC"].values.tolist()
    ACC_MCC = dict(zip(*np.unique(ACC_MCC_list, return_counts=True)))
    TP = ACC_MCC["TP"]
    TN = ACC_MCC["TN"]
    FP = ACC_MCC["FP"]
    FN = ACC_MCC["FN"]
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    return ACC, MCC, SN, SP, forest

def Random_forest_prediction(forest, test_file):
    data_test = pd.read_csv(test_file, index_col=0)
    test_independent_x = data_test.drop(["label"], axis="columns")
    test_independent_y = data_test["label"]
    test_independent_y = test_independent_y.astype(int)
    result = forest.predict(test_independent_x)
    accuracy_score(test_independent_y, result)
    newdata_test = {"name": list(test_independent_x.index),
                    "true": list(test_independent_y),
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
    return ACC, MCC, SN, SP

def main():
    # Train Random Forest
    train_file="./data/training_set_2.csv"
    acc_train, mcc_train, sn_train, sp_train, trained_forest = TrainData(train_file)

    print("Training Metrics:")
    print("ACC:", acc_train)
    print("MCC:", mcc_train)
    print("SN:", sn_train)
    print("SP:", sp_train)

    # Predict on Independent Test
    test_file = "./data/independent_test.csv"
    acc_test, mcc_test, sn_test, sp_test = Random_forest_prediction(trained_forest, test_file)

    print("\nTest Metrics:")
    print("ACC:", acc_test)
    print("MCC:", mcc_test)
    print("SN:", sn_test)
    print("SP:", sp_test)

    # Plot ROC curve for Independent Test
    data_test = pd.read_csv(test_file, index_col=0)
    test_independent_x = data_test.drop(["label"], axis="columns")
    test_independent_y = data_test["label"].astype(int)
    y_test_pred_prob = trained_forest.predict_proba(test_independent_x)[:, 1]
    acu_curve(test_independent_y, y_test_pred_prob)

if __name__ == "__main__":
    main()
