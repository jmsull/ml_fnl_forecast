import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

import shap
from shap import TreeExplainer


def permutation_importance(model, train, test, features):
    perm_importance = permutation_importance(
        model, test[features], test["b_phi"])
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(train[features].columns[sorted_idx],
             perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    # plt.savefig("Permutation_importance.pdf", bbox_inches='tight')


def SHAP_importance(model, train, test, features, names_to_plot):
    exp = TreeExplainer(model)
    shap_vals = exp.shap_values(test[features])
    shap_df = pd.DataFrame(shap_vals, columns=pd.Index(
        names_to_plot, name='features'))
    shap.summary_plot(shap_vals, test[features], plot_type="bar",
                      feature_names=names_to_plot, show=False, plot_size=(4.5, 4.5))
    plt.xlabel(r"$\overline{|SHAP|}$")
    # plt.xlabel(r"$\frac{1}{n}\sum_{i=1}^{n}|{SHAP}_i|$")
    # plt.savefig("SHAP_importance.pdf", bbox_inches='tight')


def x1_x2(pred, act, b_1):
    """Split into tertilles."""
    res_onlyb = pd.DataFrame(np.array([pred, act, b_1]).T, columns=[
                             "pred_b", "b", "b_1"])  # add b_1
    res_onlyb.index = range(res_onlyb.shape[0])
    sorted_indexes = np.argsort(res_onlyb["pred_b"].values)  # based on mass

    res_c1 = res_onlyb.iloc[sorted_indexes[:len(sorted_indexes)//3]]
    res_c2 = res_onlyb.iloc[sorted_indexes[len(
        sorted_indexes)//3: 2*len(sorted_indexes)//3]]
    res_c3 = res_onlyb.iloc[sorted_indexes[-len(sorted_indexes)//3:]]

    b1_avg1 = np.mean(res_c1["b_1"])
    b1_avg2 = np.mean(res_c2["b_1"])
    b1_avg3 = np.mean(res_c3["b_1"])

    avg1_act = np.mean(res_c1["b"])
    avg2_act = np.mean(res_c2["b"])
    avg3_act = np.mean(res_c3["b"])

    # This is the same as the values in the ''avg_act'' block
    avg1 = np.mean(res_c1["pred_b"])
    avg2 = np.mean(res_c2["pred_b"])
    avg3 = np.mean(res_c3["pred_b"])

    bb = np.array(act)
    sorted_indexes = np.argsort(bb)  # based on mass
    mean_act1 = np.mean(bb[sorted_indexes[:len(sorted_indexes)//3]])
    mean_act2 = np.mean(
        bb[sorted_indexes[len(sorted_indexes)//3:2*len(sorted_indexes)//3]])
    mean_act3 = np.mean(bb[sorted_indexes[-len(sorted_indexes)//3:]])

    b1_mean_act1 = np.mean(b_1[sorted_indexes[:len(sorted_indexes)//3]])
    b1_mean_act2 = np.mean(
        b_1[sorted_indexes[len(sorted_indexes)//3:2*len(sorted_indexes)//3]])
    b1_mean_act3 = np.mean(b_1[sorted_indexes[-len(sorted_indexes)//3:]])

    x1 = [avg1_act, avg2_act, avg3_act]
    x2 = [mean_act1, mean_act2, mean_act3]

    x1_b1 = [b1_avg1, b1_avg2, b1_avg3]
    x2_b1 = [b1_mean_act1, b1_mean_act2, b1_mean_act3]

    return x1, x2, x1_b1, x2_b1
