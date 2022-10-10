from sklearn.datasets import load_svmlight_file,dump_svmlight_file, make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import time

def experiment(dataset, total_budget, n_trees, boost_method = 'DPBoost_2level', inner_boost_round=50, balance_partition=1):
    x, y = load_svmlight_file(dataset)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(y.reshape(-1,1))
    y_new = scaler.transform(y.reshape(-1,1))
    bagging_freq = 1
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_error',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_iterations': n_trees,
        'my_n_trees': n_trees,
        'lambda_l2': 0.1,
        'bagging_freq': bagging_freq,
        'bagging_fraction':0.5,
        'max_bin': 255,
        'boost_from_average': False,
        'total_budget': total_budget,
        'boost_method': boost_method,
        'high_level_boost_round': 1,
        'inner_boost_round': inner_boost_round,
        'balance_partition': balance_partition,
        'geo_clip': 1,
        'verbose': -1,
    }
    data = lgb.Dataset(x,y_new.reshape(-1))
    results = lgb.cv(params, data, num_boost_round = n_trees, nfold = 5, stratified= False)
    print("error mean:", results["binary_error-mean"][n_trees - 1])
    print("error std:", results["binary_error-stdv"][n_trees - 1])

    return results["binary_error-mean"], results["binary_error-stdv"]


dataset_root_path = "./"


datasets = {
    "a9a"
}

def try_DPBoost_2level(output_path="output.txt", n_trees_list = [50], total_budgets_list = [1,2,4,6,8,10], inner_boost_round_list = [50]):
    with open(output_path, 'w') as output:
        for dataset in datasets:
            dataset_path = dataset_root_path + dataset
            output.write(dataset + "\n")
            for n_trees in n_trees_list:
                for total_budget in total_budgets_list:
                    print("n_trees=" + str(n_trees))
                    output.write("n_trees="+str(n_trees) + "\n")
                    output.write("total_budget="+str(total_budget)+"\n")
                    for inner_boost_round in inner_boost_round_list:
                        print("inner_boost_round="+str(inner_boost_round))
                        output.write("inner_boost_round="+str(inner_boost_round)+"\n")
                        for balance_partition in [0]:
                            start = time.time()
                            mean, stdv = experiment(dataset_path, total_budget=total_budget, n_trees=n_trees, inner_boost_round = inner_boost_round, balance_partition = balance_partition)
                            end=time.time()
                            output.write("total time="+str(end-start)+"\n")
                            output.write("mean="+str(mean)+"\n")
                            output.write("stdv="+str(stdv)+"\n")


try_DPBoost_2level()


