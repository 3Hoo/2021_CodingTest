import os
from numpy.lib.npyio import load
import torch
import numpy as np


'''
주어진 데이터를 특정 목적(shuffle 등)에 맞게 설정하기 위한 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np array) 
    - index_seq     : 주어진 데이터 list들을 어떻게 섞을지 결정하는 index가 적힌 list
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    - is_img        : 데이터가 이미지 형식인지에 대한 bool
    
[ outputs ]
    - cur_pos       : 모든 데이터의 현시점 위치 정보 (np array)
    - next_pos      : 모든 데이터의 다음 step 위치 정보 (np array)
    - inputs        : 모델의 input으로 들어갈 전체 데이터 (np array)
    - targets       : 모델의 output과 비교할 전체 label 데이터 (dX, dY) (np array)
'''
def sample_train_data(data, index_seq, use_angle_v=False, is_img=False) : 
    total_len = len(data)
    idxs = index_seq
    
    inputs = []
    targets = []
    cur_pos = []
    next_pos = []
    
    for idx in idxs : 
        pos, (tar_posX, tar_posY), input_data, (tar_dX, tar_dY) = data[idx]
        if is_img : 
            inputs.append(input_data)
        else :
            dirX = input_data[:, 0]
            dirY = input_data[:, 1]
            linX = input_data[:, 2]
            linY = input_data[:, 3]
            if use_angle_v : 
                angX = input_data[:, 4]
                angY = input_data[:, 5]
                angZ = input_data[:, 6]
                inputs.append([dirX, dirY, linX, linY, angX, angY, angZ])
            else :
                angX = None
                angY = None
                angZ = None
                inputs.append([dirX, dirY, linX, linY]) 
        targets.append([tar_dX, tar_dY])
        cur_pos.append(pos)
        next_pos.append([tar_posX, tar_posY])
        
    inputs = np.array(inputs)
    targets = np.array(targets)
    cur_pos = np.array(cur_pos)
    next_pos = np.array(next_pos)
    
    return cur_pos, next_pos, inputs, targets
      

'''
주어진 데이터를 Dsa model에 맞도록 셈플링하는 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np array) 
    - index_seq     : 주어진 데이터 list들을 어떻게 섞을지 결정하는 index가 적힌 list
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    
[ outputs ]
    - cur_pos       : 모든 데이터의 현시점 위치 정보 (np array)
    - next_pos      : 모든 데이터의 다음 step 위치 정보 (np array)
    - prev_differ   : (time_step - 1) 만큼의 데이터의 pos 변화량 = targets의 0:t-1  (np array)
    - inputs        : 모델의 input으로 들어갈 전체 데이터 (np array)
    - targets       : 모델의 output과 비교할 전체 label 데이터 (np array)
'''
def sample_train_data_DSA(data, index_seq, use_angle_v=False) :
    cur_pos, next_pos, inputs, targets = sample_train_data(data, index_seq, use_angle_v)
    #print("cur pos ", cur_pos.shape)
    print("---------", inputs.shape)
    prev_differ = []
    for i in range(len(cur_pos)) : 
        tmp = []
        for t in range(1, cur_pos.shape[-1]) :
            prev_dX = cur_pos[i,0,t] - cur_pos[i,0,t-1]
            prev_dY = cur_pos[i,1,t] - cur_pos[i,1,t-1]
            tmp.append((prev_dX, prev_dY))
        prev_differ.append(tmp)
    prev_differ = np.array(prev_differ)     # (14000,99,2)
    
    return cur_pos, next_pos, prev_differ, inputs, targets
 
    
'''
주어진 데이터를 Transformer Base model에 맞도록 셈플링하는 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np array) 
    - index_seq     : 주어진 데이터 list들을 어떻게 섞을지 결정하는 index가 적힌 list
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    
[ outputs ]
    - dir_x         : 이전 time_step 만큼의 x 방향 정보                (np array)
    - dir_y         : 이전 time_step 만큼의 y 방향 정보                (np array)
    - lin_x         : 이전 time_step 만큼의 x 방향의 선형 속도 정보     (np array)
    - lin_y         : 이전 time_step 만큼의 y 방향의 선형 속도 정보     (np array)
    - prev_dx       : 이전 time_step 만큼의 데이터의 posX의 변화량      (np array)
    - prev_dy       : 이전 time_step 만큼의 데이터의 posY의 변화량      (np array)
    - cur_pos       : 이전 time_step 만큼의 카트의 위치 정보            (np array)
    - next_pos      : 다음 time_step에서의 카트의 위치 정보             (np array)
    - targets       : 모델이 예측해야 할 target posX, posY 변화량       (np array)
'''
def sample_train_data_TF(data, index_seq, use_angle_v=False) : 
    cur_pos, next_pos, inputs, targets = sample_train_data(data, index_seq, use_angle_v)
    prev_dx = []
    prev_dy = []
    last_posX = 0
    last_posY = 0
    for i in range(len(cur_pos)) : 
        tmp_x= []
        tmp_y = []
        tmp_x.append(cur_pos[i,0,0]-last_posX)
        tmp_y.append(cur_pos[i,1,0]-last_posY)
        for t in range(1, cur_pos.shape[-1]) :
            prev_dX = cur_pos[i,0,t] - cur_pos[i,0,t-1]
            prev_dY = cur_pos[i,1,t] - cur_pos[i,1,t-1]
            tmp_x.append(prev_dX)
            tmp_y.append(prev_dY)
        last_posX = cur_pos[i,0,-1]
        last_posY = cur_pos[i,1,-1]
        prev_dx.append(tmp_x)
        prev_dy.append(tmp_y)
    prev_dx = np.array(prev_dx).reshape(-1,30,1)      # (14000,30,1) 
    prev_dy = np.array(prev_dy).reshape(-1,30,1)

    # inputs : (14000,4,30)
    dir_x = inputs[:,0,:].reshape(-1,30,1)            # (14000,30,1) 
    dir_y = inputs[:,1,:].reshape(-1,30,1)            # (14000,30,1) 
    lin_x = inputs[:,2,:].reshape(-1,30,1)            # (14000,30,1) 
    lin_y = inputs[:,3,:].reshape(-1,30,1)            # (14000,30,1) 
    
    # TF base 모델에는 (prev_dx, dir_x, dir_y, lin_x, lin_y) 또는 (prev_dy, dir_x, dir_y, lin_x, lin_y)가 input으로 들어간다
    # target은 DSA와 동일
    
    return dir_x, dir_y, lin_x, lin_y, prev_dx, prev_dy, cur_pos, next_pos, targets
    
    
    
'''
주어진 데이터를 특정 목적(shuffle 등)에 맞게 설정하여 데이터를 로드하는 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np ndarray) 
    - batch_size    : 미니배치 크기 (int)
    - device        : CPU or GPU 
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    - shuffle       : 주어진 데이터를 섞을 것인지 결정하는 bool
    - is_img        : 데이터가 이미지 형식인지에 대한 bool
    
[ outputs - yield ]
    - c             : input data 시점에서의 현재 위치       (np array)
    - n             : input data 다음 시점에서의 위치       (np array)
    - x             : 모델의 input으로 들어갈 데이터        (torch float tensor)
    - t             : 모델의 output과 비교할 label 데이터   (torch long tensor)
'''  
def train_data_loader(data, batch_size, device, use_angle_v=False, shuffle=False, is_img=False) : 
    total_len = len(data)
    idxs = np.arange(total_len)
    if shuffle : 
        np.random.shuffle(idxs)
    
    cur_pos, next_pos, inputs, targets = sample_train_data(data, idxs, use_angle_v, is_img)
    
    for start_idx in range(0, total_len, batch_size) :
        end_idx = start_idx + batch_size
        
        x = inputs[start_idx:end_idx]
        t = targets[start_idx:end_idx]
        c = cur_pos[start_idx:end_idx]
        n = next_pos[start_idx:end_idx]
        
        if device == "cuda" : 
            x = torch.Tensor(x).float().cuda()
            t = torch.Tensor(t).float().cuda()
        else :
            x = torch.Tensor(x).float()
            t = torch.Tensor(t).float()
        
        yield c, n, x, t
        

'''
주어진 데이터를 특정 목적(shuffle 등)에 맞게 설정하여 데이터를 로드하는 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np ndarray) 
    - batch_size    : 미니배치 크기 (int)
    - device        : CPU or GPU 
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    - shuffle       : 주어진 데이터를 섞을 것인지 결정하는 bool
    
[ outputs - yield ]
    - c             : input data 시점에서의 현재 위치                   (np array)
    - n             : input data 다음 시점에서의 위치                   (np array)
    - p             : input data 시점에서 -99 step 까지의 pos 변화량    (torch float tensor)
    - x             : 모델의 input으로 들어갈 데이터                    (torch float tensor)
    - t             : 모델의 output과 비교할 label 데이터               (torch long tensor)
'''  
def train_data_loader_DSA(data, batch_size, device, use_angle_v=False, shuffle=False) : 
    total_len = len(data)
    idxs = np.arange(total_len)
    if shuffle : 
        np.random.shuffle(idxs)
    
    cur_pos, next_pos, prev_differ, inputs, targets = sample_train_data_DSA(data, idxs, use_angle_v)
    
    for start_idx in range(0, total_len, batch_size) :
        end_idx = start_idx + batch_size
        
        x = inputs[start_idx:end_idx]
        t = targets[start_idx:end_idx]
        p = prev_differ[start_idx:end_idx]  # (batch, time_step-1, 2)
        c = cur_pos[start_idx:end_idx]
        n = next_pos[start_idx:end_idx]
        
        if device == "cuda" : 
            x = torch.Tensor(x).float().cuda()
            t = torch.Tensor(t).float().cuda()
            p = torch.Tensor(p).float().cuda()
        else :
            x = torch.Tensor(x).float()
            t = torch.Tensor(t).float()
            p = torch.Tensor(p).float()
        
        yield c, n, p, x, t      
        

'''
주어진 데이터를 특정 목적(shuffle 등)에 맞게 설정하여 데이터를 로드하는 함수

[ inputs ]
    - data          : 모델 훈련에 사용할 데이터 (np ndarray) 
    - batch_size    : 미니배치 크기 (int)
    - device        : CPU or GPU 
    - use_angle_v   : 각속도 데이터를 이용할 것인지에 대한 bool
    - shuffle       : 주어진 데이터를 섞을 것인지 결정하는 bool
    
[ outputs - yield ]
    - c             : input data 시점에서의 현재 위치                   (np array)
    - n             : input data 다음 시점에서의 위치                   (np array)
    - t             : 모델의 output과 비교할 label 데이터               (torch long tensor)
'''  
def train_data_loader_TF(data, batch_size, device, use_angle_v=False, shuffle=False) :
    total_len = len(data)
    idxs = np.arange(total_len)
    if shuffle : 
        np.random.shuffle(idxs)
    dir_X, dir_Y, lin_X, lin_Y, prev_dX, prev_dY, cur_pos, next_pos, targets = sample_train_data_TF(data, idxs, use_angle_v)
    
    
    for start_idx in range(0, total_len, batch_size) : 
        end_idx = start_idx + batch_size
        
        prev_dx = prev_dX[start_idx:end_idx]
        prev_dy = prev_dY[start_idx:end_idx]
        dir_x = dir_X[start_idx:end_idx]
        dir_y = dir_Y[start_idx:end_idx]
        lin_x = lin_X[start_idx:end_idx]
        lin_y = lin_Y[start_idx:end_idx]
        c = cur_pos[start_idx:end_idx]
        n = next_pos[start_idx:end_idx]
        t = targets[start_idx:end_idx]
        
        if device == "cuda" : 
            prev_dx = torch.Tensor(prev_dx).float().cuda()
            prev_dy = torch.Tensor(prev_dy).float().cuda()
            dir_x = torch.Tensor(dir_x).float().cuda()
            dir_y = torch.Tensor(dir_y).float().cuda()
            lin_x = torch.Tensor(lin_x).float().cuda()
            lin_y = torch.Tensor(lin_y).float().cuda()
            c = torch.Tensor(c).float().cuda()
            n = torch.Tensor(n).float().cuda()
            t = torch.Tensor(t).float().cuda()
        else :
            prev_dx = torch.Tensor(prev_dx).float()
            prev_dy = torch.Tensor(prev_dy).float()
            dir_x = torch.Tensor(dir_x).float()
            dir_y = torch.Tensor(dir_y).float()
            lin_x = torch.Tensor(lin_x).float()
            lin_y = torch.Tensor(lin_y).float()
            c = torch.Tensor(c).float()
            n = torch.Tensor(n).float()
            t = torch.Tensor(t).float()
            
        yield prev_dx, prev_dy, dir_x, dir_y, lin_x, lin_y, c, n, t
        
if __name__ == "__main__" : 
    import pickle
    TRAIN_DATA_PATH = "../data/1224/train_30step_data.p"

    with open(TRAIN_DATA_PATH, "rb") as p :
        TRAIN_DATA = pickle.load(p)
    
    #loader = train_data_loader_DSA(TRAIN_DATA, 10, "cuda")
    loader = train_data_loader_TF(TRAIN_DATA, 10, "cuda")
    
    for self_idx, (dx, dy, dirx, diry, linx, liny, c, n, t) in enumerate(loader) : 
        #print(c.shape)  # 10, 2, 100
        #print(n.shape)  # 10, 2
        #print(p.shape)  # 10, 99, 2
        #print(x.shape)  # 10, 4, 100
        #print(t.shape)  # 10, 2, 100
        #torch.save(p, "./p.pt")
        print(dx.shape)  # 10, 30, 1
        print(dy.shape)  # 10, 30, 1
        print(dirx.shape)  # 10, 30, 1
        print(diry.shape)  # 10, 30, 1
        print(linx.shape)  # 10, 30, 1
        print(liny.shape)  # 10, 30, 1
        print(c.shape)  # 10, 2, 30
        print(n.shape)  # 10, 2
        print(t.shape)  # 10, 2
        torch.save(dx, "./test_data/tf/dx.pt")
        torch.save(dy, "./test_data/tf/dy.pt")
        torch.save(dirx, "./test_data/tf/dirx.pt")
        torch.save(diry, "./test_data/tf/diry.pt")
        torch.save(linx, "./test_data/tf/linx.pt")
        torch.save(liny, "./test_data/tf/liny.pt")
        torch.save(c, "./test_data/tf/c.pt")
        torch.save(n, "./test_data/tf/n.pt")
        torch.save(t, "./test_data/tf/t.pt")
        break
        
    