# coding:utf-8
import matplotlib.pyplot as plt
from math import pow
import numpy as np
import pandas as pd
import time

def matrix_factorization(rank_matrix, P, Q, K, steps = 10000, alpha = 0.05, beta = 0.02):
    Q = Q.T
    result = []
    print len(rank_matrix)
    print len(rank_matrix[1])
    for step in range(steps):
        print("step: %d" % step)
        for i in range(len(rank_matrix)):
            for j in range(len(rank_matrix[i])):
                if rank_matrix[i][j] > 0:
                    eij = rank_matrix[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(rank_matrix)):
            for j in range(len(rank_matrix[i])):
                if rank_matrix[i][j] > 0:
                    e = e + pow(rank_matrix[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        result.append(e)
        print("loss: %.4f" % e)
        if e < 0.001:
            break
    return P, Q.T, result

train_file = "../datasets/train_data_5k.csv"
test_file = "../datasets/test_data_5k.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
users = 0
items = 0
train_list = train_data.values.tolist()
for row in train_list:
    if row[0] > users:
        users = row[0]
    if row [1] > items:
        items = row[1]
test_list = test_data.values.tolist()
for row in test_list:
    if row[0] > users:
        users = row[0]
    if row [1] > items:
        items = row[1]


R = np.zeros([users, items], dtype = float)
print users
print items
for row in train_list:
    if row[2] > 0.0:
        R[row[0] - 1][row[1] - 1] = row[2]



# R=[[5,3,0,1],
#    [4,0,0,1],
#    [1,1,0,5],
#    [1,0,0,4],
#    [0,1,5,4]]
# print type(R)
R = np.array(R)
# print R
N = len(R)
M = len(R[0])
K = 10
P = np.random.rand(N, K)
Q = np.random.rand(M, K)
start_time = time.time()
nP, nQ, result = matrix_factorization(R, P, Q, K)
end_time = time.time()
duration = end_time - start_time
print("运行时间为：" )
print duration
print("原始的评分矩阵R为: ")
print R
R_MF = np.dot(nP, nQ.T)
print("预测结果矩阵R_MF为: ")
print R_MF
n=len(result)
x=range(n)
plt.plot(x,result,color='r',linewidth=3)
plt.title("Convergence curve")
plt.xlabel("generation")
plt.ylabel("loss")
plt.show()
pred =  pd.DataFrame(R_MF.tolist())
pred.to_csv("../datasets/MF_pred.csv", index = False, encoding = "utf8")