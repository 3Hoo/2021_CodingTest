# data_analyze결과, posX와 posY 데이터에 직접적으로 연관되어있는 데이터는 linear Velocity와 direction 임을 알 수 있었다
# 단, Z 축과 관련된 데이터의 경우 p_value로 보나 correlation으로 보나 posX와 posY에 직접적인 관계를 맺고 있음을 알 수 없으므로 Z축 데이터는 제외한다
# 단, angle Velocity의 경우 확실히 결정하기 힘들므로 (posX는 완전 관계없는 것처럼 보이지만, posY와는 어느정도 관계가 있는 것 같이 보인다)
# 데이터를 두 가지 버전으로 나눈다
# -- 1) angle Velocity 제외
# -- 2) angle Velocity 포함 


########################
#       설   정        #
########################
# 1) time step length는 0.01s의 100배인 1s으로 설정하여 신경망 모델이 지난 1s의 데이터를 보고 앞의 0.01s 의 결과를 예측하도록 한다
# 2) dirX, dirY, linX, linZ, angX, angY, angZ 각각 0 ~ 1의 값을 가지도록 normalize 한다
#    맵의 크기는 정해져있으므로 posX와 posY를 normalize해도 실제 주행에 사용할 수 있도록 변환할 수 있지만, 일단 여기서는 제외하고 해보자.
#    우선은 모델의 input으로 들어가지도 않고, 더 세밀하거나 극적인 변화에 대응할 수 있도록 유도할 수 있을 것이라고 생각한다.
# 3) target 값은 현재(1s의 데이터 중 가장 마지막) posX, posY와 0.01s 이후(1.01s)의 posX, posY의 차이가 된다.
#    최종 결과가 좀 더 물리적으로 올바를 수 있도록 모델이 최종 위치를 계산하는 그래프를 탐색하는 것이 아니라, 주어진 물리량에 대한 '변화값'을 계산하는 그래프를 탐색하도록 유도하기 위함이다

# 4) 여기서는 model input으로 img 형식이 들어간다. img는 다음과 같이 만든다
#    - pos 데이터도 0~1 범위를 가지도록 normalize 한다. 단, 변화량 계산은 normalize 전에 한다 (너무 작지 않고 적당한 loss 값을 얻기 위해) 
#      실제 pos로 복원하기 위해서 MinMaxScaler도 저장한다 => 필요 없을지도? (cur pos와 next pos를 normalize 하기 전에 기록할 수 있으므로)
#    - ViT에서 사용한 아이디어를 일부 채용한다
#    - [[posX, posY, dirX], [dirY, linX, linY], [argX, argY, argZ]] 값을 가지는 3X3 블록을 만든다
#    - 위의 블록을 회전하면서 총 4개의 블록을 이어붙여 6X6 블록을 만든다
#    - 다시 위의 블록을 좌우대칭/상하대칭으로 이어붙여 6X6X3 블록을 만든다 => 이것이 하나의 이미지가 되어 모델에 들어간다


import os
import pickle
from typing_extensions import final
import numpy as np
import csv
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import joblib

csv_path = "../../data/DATA.csv"

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
            final_data.append((time, posX, posY, dirX, dirY, linearvX, linearvY, angvX, angvY, angvZ))
            
            pred_time = time
            step_list = []
            step_list.append(r)
        else :
            step_list.append(r)


final_data = np.array(final_data)
final_data = final_data.astype('float64')
print(final_data.shape)


#----------------- normalize -----------------#
for i in range(3, 10) : 
    final_data[:, i] = minmax_scale(final_data[:, i])
posX_scaler = MinMaxScaler(feature_range=(0, 1))
posY_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_posX_data = final_data[:, 1].reshape(-1, 1)
scaled_posY_data = final_data[:, 2].reshape(-1, 1)
scaled_posX_data = posX_scaler.fit_transform(scaled_posX_data)
scaled_posY_data = posY_scaler.fit_transform(scaled_posY_data)
#final_data[:, 1] = scaled_posX_data.reshape(-1)
#final_data[:, 2] = scaled_posY_data.reshape(-1)
scaled_posX_data = scaled_posX_data.reshape(-1)
scaled_posY_data = scaled_posY_data.reshape(-1)

#----------------- split data -----------------#
# 총 15416개의 데이터
# 14000개의 train data, 1000개의 val data, (416-100-1)=315개의 test data 로 split
# 전체 데이터를 1s 단위로 쪼갠 후, 인덱스 생성 => 인덱스 랜덤하게 shuffle => 이후 14000:1000:-1 으로 split 하여 저장
STEP_LENGTH = 100 # 0.01s * 100 = 1s
base_save_path = "../../data/1224_img"
start_idx = STEP_LENGTH

idxs = np.arange(start_idx, len(final_data)-1)
np.random.shuffle(idxs)
train_idxs = idxs[:14000]
val_idxs = idxs[14000:15000]
test_idxs = idxs[15000:]

#----------------- make data -----------------#

count = 0
train_data = []
for idx in train_idxs :
    current_posX = final_data[idx, 1]
    current_posY = final_data[idx, 2]
    dirX = final_data[idx, 3]
    dirY = final_data[idx, 4]
    linX = final_data[idx, 5]
    linY = final_data[idx, 6]
    angX = final_data[idx, 7]
    angY = final_data[idx, 8]
    angZ = final_data[idx, 9]
    next_posX = final_data[idx+1, 1]
    next_posY = final_data[idx+1, 2]
    target_dX = next_posX - current_posX
    target_dY = next_posY - current_posY

    # 기본 block 생성
    block_3x3 = np.zeros((3, 3))
    block_3x3[0] = np.array([scaled_posX_data[idx], scaled_posY_data[idx], dirX])
    block_3x3[1] = np.array([dirY, linX, linY])
    block_3x3[2] = np.array([angX, angY, angZ])
    
    # 기본 block2 생성
    block_6x6 = np.zeros((6, 6))
    block_6x6[0:3, 0:3] = block_3x3
    block_6x6[0:3, 3:6] = np.rot90(block_3x3, 1)
    block_6x6[3:6, 3:6] = np.rot90(block_3x3, 2)
    block_6x6[3:6, 0:3] = np.rot90(block_3x3, 3)
    
    # block2 쌓기
    img_block = np.zeros((6, 6, 3))
    img_block[:,:,0] = block_6x6
    img_block[:,:,1] = np.flip(block_6x6, axis=0)
    img_block[:,:,2] = np.flip(block_6x6, axis=1)

    current_pos = (current_posX, current_posY)  
    target_pos = (next_posX, next_posY)         # for final comparison
    target_d = (target_dX, target_dY)           # model target
    train_data.append((current_pos, target_pos, img_block, target_d))
    
#train_data = np.array(train_data)
with open(os.path.join(base_save_path, "train_data.p"), "wb") as p :
    pickle.dump(train_data, p)

val_data = []
for idx in val_idxs :
    current_posX = final_data[idx, 1]
    current_posY = final_data[idx, 2]
    dirX = final_data[idx, 3]
    dirY = final_data[idx, 4]
    linX = final_data[idx, 5]
    linY = final_data[idx, 6]
    angX = final_data[idx, 7]
    angY = final_data[idx, 8]
    angZ = final_data[idx, 9]
    next_posX = final_data[idx+1, 1]
    next_posY = final_data[idx+1, 2]
    target_dX = next_posX - current_posX
    target_dY = next_posY - current_posY

    # 기본 block 생성
    block_3x3 = np.zeros((3, 3))
    block_3x3[0] = np.array([scaled_posX_data[idx], scaled_posY_data[idx], dirX])
    block_3x3[1] = np.array([dirY, linX, linY])
    block_3x3[2] = np.array([angX, angY, angZ])
    
    # 기본 block2 생성
    block_6x6 = np.zeros((6, 6))
    block_6x6[0:3, 0:3] = block_3x3
    block_6x6[0:3, 3:6] = np.rot90(block_3x3, 1)
    block_6x6[3:6, 3:6] = np.rot90(block_3x3, 2)
    block_6x6[3:6, 0:3] = np.rot90(block_3x3, 3)
    
    # block2 쌓기
    img_block = np.zeros((6, 6, 3))
    img_block[:,:,0] = block_6x6
    img_block[:,:,1] = np.flip(block_6x6, axis=0)
    img_block[:,:,2] = np.flip(block_6x6, axis=1)

    current_pos = (current_posX, current_posY)  
    target_pos = (next_posX, next_posY)         # for final comparison
    target_d = (target_dX, target_dY)           # model target
    
    val_data.append((current_pos, target_pos, img_block, target_d))
#val_data = np.array(val_data)
with open(os.path.join(base_save_path, "val_data.p"), "wb") as p :
    pickle.dump(val_data, p)
    
test_data = []
for idx in test_idxs :
    current_posX = final_data[idx, 1]
    current_posY = final_data[idx, 2]
    dirX = final_data[idx, 3]
    dirY = final_data[idx, 4]
    linX = final_data[idx, 5]
    linY = final_data[idx, 6]
    angX = final_data[idx, 7]
    angY = final_data[idx, 8]
    angZ = final_data[idx, 9]
    next_posX = final_data[idx+1, 1]
    next_posY = final_data[idx+1, 2]
    target_dX = next_posX - current_posX
    target_dY = next_posY - current_posY

    # 기본 block 생성
    block_3x3 = np.zeros((3, 3))
    block_3x3[0] = np.array([scaled_posX_data[idx], scaled_posY_data[idx], dirX])
    block_3x3[1] = np.array([dirY, linX, linY])
    block_3x3[2] = np.array([angX, angY, angZ])
    
    # 기본 block2 생성
    block_6x6 = np.zeros((6, 6))
    block_6x6[0:3, 0:3] = block_3x3
    block_6x6[0:3, 3:6] = np.rot90(block_3x3, 1)
    block_6x6[3:6, 3:6] = np.rot90(block_3x3, 2)
    block_6x6[3:6, 0:3] = np.rot90(block_3x3, 3)
    
    # block2 쌓기
    img_block = np.zeros((6, 6, 3))
    img_block[:,:,0] = block_6x6
    img_block[:,:,1] = np.flip(block_6x6, axis=0)
    img_block[:,:,2] = np.flip(block_6x6, axis=1)

    current_pos = (current_posX, current_posY)  
    target_pos = (next_posX, next_posY)         # for final comparison
    target_d = (target_dX, target_dY)           # model target
    test_data.append((current_pos, target_pos, img_block, target_d))
#test_data = np.array(test_data)
with open(os.path.join(base_save_path, "test_data.p"), "wb") as p :
    pickle.dump(test_data, p)
    
# scaler 저장
joblib.dump(posX_scaler, os.path.join(base_save_path, "scaler", "posX_scaler.pt"))
joblib.dump(posY_scaler, os.path.join(base_save_path, "scaler", "posY_scaler.pt"))

print("DATA : (train) {} / (val) {} / (test) {}".format(len(train_data), len(val_data), len(test_data)))