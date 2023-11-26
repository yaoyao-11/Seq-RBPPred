import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

def acu_curve(y, prob):
    # Calculate ROC curve and AUC
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
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
    plt.title('Receiver operating characteristic example',
              font=Path('TNR.ttf'), fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    plt.show()

def featureSet(data):
    # Create feature set from the given data
    data_num = len(data)
    Xlist = []
    for row in range(0, data_num):
        tmp_list = []
        for ii in range(len(data.columns)):
            tmp_list.append(data.iloc[row][data.columns[ii]])
        Xlist.append(tmp_list)
    return Xlist

def train(TrainData):
    # Train the model using the training data
    data = pd.read_csv(TrainData, index_col=0)
    data = data.sample(frac=1, random_state=1)
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    train_x = featureSet(train.drop(["label"], axis="columns"))
    y_train = train["label"].values
    train_x = np.array(train_x)
    model = CascadeForestClassifier(random_state=1)
    model.fit(train_x, y_train)
    return model
def Deep_forest_prediction(model, test_file):
    # Predict using the trained model and display ROC curve
    data_test = pd.read_csv(test_file, index_col=0)
    test_indepent_x = data_test.drop(["label"], axis="columns")
    test_indepent_y = data_test["label"].astype(int)
    result = model.predict(test_indepent_x)
    y_test_pred = model.predict_proba(test_indepent_x)[:, 1]
    acu_curve(test_indepent_y, y_test_pred)
    result = np.where(result > 0.5, 1, 0)
    newdata_test = {"name": list(test_indepent_x.index),
                    "true": list(test_indepent_y),
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

# Main program
def main():
    # Train the model
    TrainData = "./data/training_set_2.csv"
    trained_model = train(TrainData)

    # Predict and display ROC curve
    test_file = "./data/independent_test.csv"
    Deep_forest_prediction(trained_model, test_file)

if __name__ == "__main__":
    main()
