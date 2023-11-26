"""
When you want to obtain ./Dmlab/MCFS_4801_6243/test.txt, 
you must first execute mcfs.run, 
and you can choose a custom name for MCFS_4801_6243.
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams

def read_data(test_file, rules_file):
    data = pd.read_csv(test_file)
    p = 1626
    n = 1956 - 1627
    cls = ["Positive"] * p + ["Negative"] * n
    data = data.set_index("Unnamed: 0")
    newdata = data.copy()
    newdata["pre"] = cls
    return newdata

def apply_rules(newdata, rules_file):
    for name in newdata.index:
        a = 0
        with open(rules_file, "r") as f:
            for i in f:
                i = i.replace("\n", "")
                a += 1
                data_name = "pre_{}".format(a)
                feature_value = []
                judge = []
                judge_value = []
                for features in i.split("=>")[0].split("and"):
                    feature = features.lstrip().replace("(", "").replace(")", "").split(" ")[0]
                    symbol = features.lstrip().replace("(", "").replace(")", "").split(" ")[1]
                    need_value = features.lstrip().replace("(", "").replace(")", "").split(" ")[2]
                    need_value = float(need_value)
                    feature_value.append(newdata.loc[name, feature])
                    judge.append(symbol)
                    judge_value.append(need_value)
                num = len(feature_value)
                bool_list = []
                and_num = 0
                while and_num < num:
                    bool_v = eval(f'({feature_value[and_num]} {judge[and_num]} {judge_value[and_num]} )')
                    bool_list.append(bool_v)
                    and_num += 1
                if np.array(bool_list).all():
                    newdata.loc[name, data_name] = 'Positive'
                else:
                    newdata.loc[name, data_name] = 'Negative'

def generate_final_predictions(newdata):
    for i in newdata.index:
        pre_last = []
        for num in range(1, a + 1):
            if newdata.loc[i, "pre_{}".format(num)] == "Positive":
                pre_last.append("True")
        if len(pre_last) > 0:
            newdata.loc[i, "last"] = "Positive"
        else:
            newdata.loc[i, "last"] = "Negative"

    for i in newdata.index:
        if newdata.loc[i, "pre"] == newdata.loc[i, "last"]:
            newdata.loc[i, "prediction"] = "T"
        else:
            newdata.loc[i, "prediction"] = "F"

def calculate_metrics(newdata):
    ACC_MCC = ["NULL"] * (n + p)
    newdata["ACC_MCC"] = ACC_MCC

    for i in range(p):
        if newdata.loc[i, "prediction"] == "T":
            newdata.loc[i, "ACC_MCC"] = "TP"
        else:
            newdata.loc[i, "ACC_MCC"] = "FN"

    for i in range(p, p + n):
        if newdata.loc[i, "prediction"] == "T":
            newdata.loc[i, "ACC_MCC"] = "TN"
        else:
            newdata.loc[i, "ACC_MCC"] = "FP"

    prediction = newdata["prediction"].values.tolist()
    T_F = dict(zip(*np.unique(prediction, return_counts=True)))
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

    return ACC, MCC, SN, SP

def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
 
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('Receiver operating characteristic example', fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    
    plt.show()

def main():
    test_file = "./data/independent_test.csv"
    rules_file = "./Dmlab/MCFS_4801_6243/test.txt"
    newdata = read_data(test_file, rules_file)
    apply_rules(newdata, rules_file)
    generate_final_predictions(newdata)
    ACC, MCC, SN, SP = calculate_metrics(newdata)
    print(f"Accuracy: {ACC}")
    print(f"MCC: {MCC}")
    print(f"Sensitivity: {SN}")
    print(f"Specificity: {SP}")

    # Assuming you have the necessary data for acu_curve function
    data_test = pd.read_csv("./dmlab.csv", index_col=0)
    y_test_pred = list(data_test["pre"])
    test_independent_y = list(data_test["class"])
    acu_curve(test_independent_y, y_test_pred)

if __name__ == "__main__":
    main()
