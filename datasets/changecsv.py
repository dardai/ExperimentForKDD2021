import random
import pandas as pd
import numpy as np

#随机提取1.3w数据，然后筛选掉交互记录不足的，成为1w量级，然后接着提取出数据，筛选出5k，接着筛选，以此类推

#从path1数据中随机提取num1~num2个数中随机个到path2中
def get_data(path1, path2, num) :
    file1 = pd.read_csv(path1)
    old_file = open(path1, 'r', encoding='UTF-8')
    new_file = open(path2, 'w', encoding='UTF-8')
    new_file.truncate() #清空文件
    len_file1 = len(file1.userid.values[:]) #计算长度

    scale = list(range(1, len_file1))
    resultList = random.sample(scale, num if len(scale) > num else len(scale))
    columns = file1.columns.values

    lines = old_file.readlines()
    new_file.write(lines[0])
    for i in resultList:
        new_file.write(lines[i])

#筛选出的数据，将交互条数少于num的删除掉，最后保存到csv中
def remove_data(path, num):
    df = pd.read_csv(path)  # 导入文件
    print("删除前：")
    print(df.describe()) # 删除前整体概况
    ts = df['userid'].value_counts() # 按照userid筛选数量

    for i in ts.index:
        if ts[i] < num:
            df = df[df['userid'] != i]
    df.to_csv(path,index=0)
    df = pd.read_csv(path)
    print("删除后：")
    print(df.describe())  # 删除完后整体概况

if __name__ == '__main__':
    #提取出1w量级
    path1 = "E:/data/data.csv"
    path2 = "E:/data/data_1w.csv"
    num = 13000
    get_data(path1,path2,num)
    remove_data(path2, 10)

    #提取出5k量级
    path3 = "E:/data/data_5k.csv"
    num = 6000
    get_data(path2, path3, num)
    remove_data(path3, 5)

    #提取出1k量级
    path4 = "E:/data/data_1k.csv"
    num = 1300
    get_data(path3, path4, num)
    remove_data(path4, 2)





