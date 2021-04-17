# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import sklearn
print sklearn.__version__
from pyfm import pylibfm
from sklearn.metrics import mean_squared_error
from evaluators.NDCG import get_ndcg


def load_data(filename):
    data = []
    label = []
    users = set()
    items = set()
    data_pd = pd.read_csv(filename)
    data_list = data_pd.values.tolist()
    print data_list
    for row in data_list:
        data.append({"user_id": str(row[0]), "item_id": str(row[1])})
        label.append(float(row[2]))
        users.add(str(row[0]))
        items.add(str(row[1]))
    return data, np.array(label), users, items

def get_avg_ndcg(result_list, users, list_len):
    ndcg_list = []
    for user in users:
        temp_list = []
        for row in result_list:
            if row[0] == user:
                temp_list.append(row)
        # print len(temp_list)
        if len(temp_list) >= list_len:
            ndcg_list.append(get_ndcg(temp_list, list_len))
    sum = 0.0
    for i in range(len(ndcg_list)):
        sum = sum + ndcg_list[i]
    avg_ndcg = sum / float(len(ndcg_list))
    return avg_ndcg

train_file = "../datasets/train.csv"
test_file = "../datasets/test.csv"
train_data, train_y, train_users, train_items = load_data(train_file)
test_data, test_y, test_users, test_items = load_data(test_file)
v = DictVectorizer()
train_x = v.fit_transform(train_data)
test_x = v.transform(test_data)

fm = pylibfm.FM(num_factors = 10,
                num_iter = 100,
                verbose = True,
                task = "regression",
                initial_learning_rate = 0.001,
                learning_rate_schedule = "optimal")
fm.fit(train_x, train_y)
preds = fm.predict(test_x)
result = []
for row in test_data:
    temp = []
    temp.append(row["user_id"])
    temp.append(row["item_id"])
    result.append(temp)
for i in range(len(preds)):
    result[i].append(preds[i])
    result[i].append(test_y[i])
print("FM MSE: %.4f" % mean_squared_error(test_y, preds))
print preds
print(result)
test_users = list(test_users)
list_len = 20
ndcg = get_avg_ndcg(result, test_users, list_len)
print ndcg
result = pd.DataFrame(result)
result.to_csv("../datasets/FM_pred.csv")