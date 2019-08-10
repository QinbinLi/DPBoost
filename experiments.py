from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

def experiment_binary(dataset, total_budget, n_trees):
    x,y=load_svmlight_file(dataset)
    params = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric': 'binary_error',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'num_iterations': n_trees,
        'lambda_l2': 0.1,
        'bagging_freq': 0,
        'max_bin': 255,
        'boost_from_average': False,
        'total_budget': total_budget,
    }
    data = lgb.Dataset(x,y)
    results = lgb.cv(params, data, num_boost_round = n_trees, nfold = 5)
    print("error mean:", results["binary_error-mean"][n_round - 1])
    print("error std:", results["binary_error-stdv"][n_round - 1])

def experiment_regression(dataset, total_budget, n_trees):
    x,y=load_svmlight_file(dataset)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(y.reshape(-1,1))
    y_new = scaler.transform(y.reshape(-1,1))
    params = {
        'boosting_type': 'goss',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'num_iterations': n_trees,
        'lambda_l2': 0.1,
        'bagging_freq': 0,
        'total_budget': total_budget,
    }
    data = lgb.Dataset(x,y_new.reshape(-1))
    results = lgb.cv(params, data, num_boost_round = n_trees, nfold = 5, stratified= False)
    multiplier = (scaler.data_max_ - scaler.data_min_) / 2
    scale_mean = results["rmse-mean"][n_round - 1] * multiplier
    scale_stdv = results["rmse-stdv"][n_round - 1] * multiplier

    print("rmse mean:", scale_mean)
    print("rmse std:", scale_stdv)


experiment_binary("datasets/a9a", total_budget=200, n_trees = 500)
# experiment_regression("datasets/YearPredictionMSD")
