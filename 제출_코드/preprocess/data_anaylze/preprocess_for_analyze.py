import os
import csv
import numpy as np
import math
import pickle

csv_path = "../../data/DATA.csv"

# time 축이 0.01 단위가 아니므로 0.01초 단위로 데이터를 묶은 후에 평균을 낸다

# [ 12/22 ] 
# => 1) 플레이어가 버튼을 누르는 것에 대해 카트의 방향/속도/위치가 종속적이라고 할 수 있다
# => 2) 모델의 output은 플레이어의 행동을 예측하는 것이 아니라, 카트의 위치를 예측하는 것이다
# => 3) 카트의 위치는 카트의 이전 step들의 방향/속도/위치에 영향을 받을 것이다
# => 4) 1의 가정으로 인해 카트의 이전 step들의 방향/속도/위치 데이터에 플레이어의 버튼 입력 데이터가 포함되어있다고 할 수 있다
# ====> 따라서 플레이어의 버튼 입력 데이터는 제외한다
# => 5) Z축과 관련된 정보는 어떻게 해야 할지 판단이 서지 않으므로, 최종 output이 될 X.pos, Y.pos 데이터와의 상관계수를 측정하여 데이터 상관관계를 파악한 후 판단해보자

final_data = []
cnt = 0
with open(csv_path, newline='') as csvf : 
    kart_data = csv.reader(csvf, delimiter = ',', quotechar='|')
    step_list = []
    pred_time = 3.12
    for row in kart_data :
        # time, posX, posY, posZ, dirX, dirY, dirZ, linearvX, linearvY, linearvZ, angvX, angvY, angvZ = row[0:13]
        r = row[0:13]
        time = round(float(r[0]), 2)
        # time의 소숫점 3번째에서 반올림하여 0.01s 단위로 변형
        # - 이전 스텝보다 증가했으면 step_list의 값들을 열 단위로 평균을 내고, 결과로 나온 1d array를 최종 data로 저장
        if time > pred_time : 
            # 전부 str 타입이므로 numpy float64 type으로 형변환
            if len(step_list) <= 1 : 
                cnt+=1
            step_list = np.array(step_list)
            step_list = step_list.astype('float64')
            this_step = np.mean(step_list, axis=0)
            (_, posX, posY, posZ, dirX, dirY, dirZ, linearvX, linearvY, linearvZ, angvX, angvY, angvZ) = this_step
            final_data.append((time, posX, posY, posZ, dirX, dirY, dirZ, linearvX, linearvY, linearvZ, angvX, angvY, angvZ))
            
            pred_time = time
            step_list = []
            step_list.append(r)
        else :
            step_list.append(r)


final_data = np.array(final_data)
final_data = final_data.astype('float64')
print(final_data.shape)
print(cnt)

with open("../data/1222/kart_data_without_normalization.p", "wb") as p : 
    pickle.dump(final_data, p)