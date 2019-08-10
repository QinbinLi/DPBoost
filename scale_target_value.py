from sklearn.datasets import load_svmlight_file,dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler

x,y=load_svmlight_file("datasets/YearPredictionMSD")
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(y.reshape(-1,1))
print(scaler.data_max_)
print(scaler.data_min_)
x_train,y_train = load_svmlight_file("datasets/YearPredictionMSD_train")
x_test,y_test=load_svmlight_file("datasets/YearPredictionMSD_test")
y_new = scaler.transform(y_train.reshape(-1,1))
dump_svmlight_file(x_train,y_new.reshape(-1),"datasets/YearPredictionMSD_scale_y.train", zero_based=False)
y_new = scaler.transform(y_test.reshape(-1,1))
dump_svmlight_file(x_test,y_new.reshape(-1),"datasets/YearPredictionMSD_scale_y.test", zero_based=False)



