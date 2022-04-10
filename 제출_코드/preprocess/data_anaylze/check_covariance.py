import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats

with open("../data/1222/kart_data_without_normalization.p", 'rb') as p :
    data = pickle.load(p)
    
posXs = data[:, 1]
posYs = data[:, 2]
posZs = data[:, 3]
dirXs = data[:, 4]
dirYs = data[:, 5]
dirZs = data[:, 6]
linXs = data[:, 7]
linYs = data[:, 8]
linZs = data[:, 9]
angXs = data[:, 10]
angYs = data[:, 11]
angZs = data[:, 12]

l = [posXs,posYs,posZs,dirXs,dirYs,dirZs,linXs,linYs,linZs,angXs,angYs,angZs]

 
cov_matrix = np.zeros(shape=(data.shape[1] - 1, data.shape[1] - 1))
cor_matrix = np.zeros(shape=(data.shape[1] - 1, data.shape[1] - 1))
p_matrix = np.zeros(shape=(data.shape[1] - 1, data.shape[1] - 1))

for i in range(len(l)) :
    for j in range(len(l)) : 
        if i == j : 
            continue
        covariance = np.cov(l[i], l[j])[0][1]           # i와 j의 공분산/상관계수
        correlation = np.corrcoef(l[i], l[j])[0][1]     # i와 j의 공분산/상관계수
        _, p_value = stats.pearsonr(l[i], l[j])         # 상관계수가 유의미한가 무의미한가
        cov_matrix[i][j] = covariance
        cor_matrix[i][j] = correlation
        p_matrix[i][j] = p_value
        
df = pd.DataFrame(cov_matrix)
df2 = pd.DataFrame(cor_matrix)
df3 = pd.DataFrame(p_matrix)
df.to_csv("1222_covariance.csv", index=False)
df2.to_csv("1222_correlation.csv", index=False)
df3.to_csv("1222_pValue.csv", index=False)