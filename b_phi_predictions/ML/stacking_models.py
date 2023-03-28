import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import svm

from DNN import *

# Not used
# import lightgbm as lgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import ElasticNet, Lasso


def train_test(data, val=True):
    """Drop the galaxies near the edges of the simulation. 
    Split data into train test, validation subsets based on 'x' spatial position. The split is 70-20-10%. 
    If val==False we return train-test split in the 80-20 ratio."""

    # Drop data near edges of simulation (units of kpc/h)
    data = data[(data["posX"] > 1_000) & (data["posX"] < 204_000) &
                (data["posY"] > 1_000) & (data["posY"] < 204_000) & (data["posZ"] > 1_000) & (data["posZ"] < 204_000)]

    # x 'legnth' of simulation now is 203k kpc/h -> make train, test, validation splits:
    test = data[data["posX"] < 40_600].copy()  # 20%
    if val == True:
        validation = data[(data["posX"] > 40_600) & (
            data["posX"] < 60_900)].copy()  # 10%
        train = data[data["posX"] > 60_900].copy()  # 70%
        return train, test, validation
    else:
        train = data[data["posX"] > 40_600].copy()  # 70%
        return train, test


def make_models(params):
    """Models to add to stacking (excluded DNN, which will be added manually). 
    Params is a dict object containing hyperparams for the models."""
    svr_model = svm.SVR(**params["svr"])  # rbf, 5
    xgboost_moel = XGBRegressor(**params["xgb"])
    # Some other models such as RF or simple Lasso/Elascti-Net/Ridge,... were tried but underperformed
    # lgb_model = lgb.LGBMRegressor(random_state=42)   # not included, xgboost better
    # lgb_model.set_params(**params["lgb"]) # drop rate
    # rf_model = RandomForestRegressor(**params["rf"]) # n_estimators, max_depth, min_samples_leaf, min_samples_split
    # ridge_model = Ridge(**params["ridge"])  # alpha
    # lasso_model = Lasso(**params["lasso"])  # alpha
    # elastic_net_model = ElasticNet(**params["elastic_net"]) # alpha
    # can be arbitrary long list of models compatible with scikit-learn
    models = [svr_model, xgboost_moel]
    return models


def stacked_model(X_train, y_train, X_test, y_test, models, params, drop_features=["posY"], batch=10, N=13):
    """Produce predictions with all base learners, as defined in 'make_models' + DNN model as defined in 'DNN.py'. """
    X_train = X_train.copy().drop(drop_features, axis=1)
    X_test = X_test.copy().drop(drop_features, axis=1)

    model1, model2 = models  # Should iterate if there are multiple models
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    # make predictions with DNN
    n_epoch = params["DNN"]["n_epoch"]  # 150
    LR, WD = 1.e-7, 1.e-7  # hardcoded, changing did not lead to improvements

    net = make_network(params["DNN"]["hidden_layers"], params["DNN"]
                       ["hidden_layer_size"], N, dropout=params["DNN"]["dropout"], bias=True)

    train_loader = convert_to_tensors(X_train, y_train, batch=batch)
    test_loader = convert_to_tensors(X_test, y_test, batch=batch)
    train_losses, valid_losses = main_training(
        net, train_loader, test_loader, LR, WD, n_epoch)
    # print("Training loss;", train_losses)
    # print("Test/validation loss;", valid_losses)
    pred_actual = validation(test_loader, net)

    pred_actual_test = []
    # Convert to appropriate type
    for batch in pred_actual:
        for i in range(len(batch[0])):
            pred_actual_test.append((float(batch[0][i]), float(batch[1][i])))

    predictions_dnn = np.array(pred_actual_test)[:, 0]
    return pd.DataFrame([predictions1, predictions2, predictions_dnn]).T


def stacked_predictions(X_train, y_train, X_test, y_test, params, drop_features=["posY"], batch=10, N=13):
    """Main function for stacking predictions, using cross validation. 'Test' can be used for both actual test set and validation set."""
    # Cross validation split is needed. Data should contain posY information! (which is dropped for training)
    models = make_models(params)

    # Hardcode this value based on the desired cross validation split
    x_min_split = 68300
    x_max_split = 136700

    X_train, y_train, X_test, y_test = X_train.copy(
    ), y_train.copy(), X_test.copy(), y_test.copy()
    # Also reset indexes to avoid any possible index errors
    X1, y1 = X_train[X_train["posY"] < x_min_split].reset_index(
        drop=True), y_train[X_train["posY"] < x_min_split].reset_index(drop=True)
    X2, y2 = X_train[X_train["posY"] <
                     x_max_split].reset_index(drop=True), y_train[X_train["posY"] < x_max_split].reset_index(drop=True)
    X2, y2 = X2[X2["posY"] > x_min_split].reset_index(
        drop=True), y2[X2["posY"] > x_min_split].reset_index(drop=True)
    X3, y3 = X_train[X_train["posY"] >
                     x_max_split].reset_index(drop=True), y_train[X_train["posY"] > x_max_split].reset_index(drop=True)

    X_merge1, y_merge1 = pd.concat(
        [X1, X2], axis=0), pd.concat([y1, y2], axis=0)
    X_merge2, y_merge2 = pd.concat(
        [X1, X3], axis=0), pd.concat([y1, y3], axis=0)
    X_merge3, y_merge3 = pd.concat(
        [X2, X3], axis=0), pd.concat([y2, y3], axis=0)
    X_merge_all, y_merge_all = pd.concat(
        [X1, X2, X3], axis=0), pd.concat([y1, y2, y3], axis=0)

    pred1 = stacked_model(X_merge1, y_merge1, X3, y3.values,
                          models, params, drop_features, batch=batch, N=N)
    pred2 = stacked_model(X_merge2, y_merge2, X2, y2.values,
                          models, params, drop_features, batch=batch, N=N)
    pred3 = stacked_model(X_merge3, y_merge3, X1, y1.values,
                          models, params, drop_features, batch=batch, N=N)
    pred_val = stacked_model(X_merge_all, y_merge_all, X_test,
                             y_test, models, params, drop_features, batch=batch, N=N)

    meta_X, meta_y = pd.concat(
        [pred1, pred2, pred3], axis=0), pd.concat([y3, y2, y1], axis=0)

    meta_learner = Ridge()
    meta_learner.fit(meta_X, meta_y)
    return meta_learner.predict(pred_val)
