'''
총 n번의 epoch동안 훈련을 진행한 후에
각 epoch 마다 저장한 모델을 로드하여 test data에 대한 err을 비교하여
가장 낮은 err을 가지는 epoch의 모델을 탐색한다
계산 결과는 json 형식으로 저장한다
'''

import numpy as np
import pickle
import os
import argparse
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.nn.functional import batch_norm

import data_manager as dm
import model
from utils import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--bUseAngleV', default=False, type=str2bool)
parser.add_argument('--epoch', type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--log_dir', type=str)

args = parser.parse_args()

assert args.model_type in ['LSTM_Attention', "DSA", 'Transformer', 'CNN', 'ViT']

# Data Load
TEST_DATA_PATH = "../../data/1224_img/test_data.p"

with open(TEST_DATA_PATH, "rb") as p :
    TEST_DATA = pickle.load(p)
    
if torch.cuda.is_available() : 
    device = "cuda"
else :
    device = "cpu"
    
model_dir = os.path.join(args.model_dir, "parm", str(args.epoch)+".pt")
summary_json = args.log_dir

time_step = 100
if args.bUseAngleV : 
    latent_dim = 7
    hidden_dim = 10
else :
    latent_dim = 4
    hidden_dim = 6
batch_size=1
    
if args.model_type == "LSTM_Attention" : 
    model_dX = model.lstm_dX(device=device, time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
    model_dY = model.lstm_dY(device=device, time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
elif args.model_type == "DSA" : 
    model_d = model.DSA(device=device, time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
elif args.model_type == "CNN" :
    model_d = model.simpleCNN(device=device)

model_d.load_state_dict(torch.load(model_dir))
model_d.cuda()
model_d.eval()

test_loader = dm.train_data_loader(TEST_DATA, batch_size=batch_size, device="cuda", use_angle_v=args.bUseAngleV, shuffle=False, is_img=True)

total_err = 0
len = 0
for self_idx, (cur_pos, next_pos, x, target) in enumerate(test_loader) : 
    dx_ = model_d(x)
    dy_ = model_d(x)
    #print(dx_.shape, cur_pos[:,0].shape, next_pos[:,0].shape)
    real_pos = np.array([next_pos[:,0], next_pos[:, 1]])
    pred_pos = np.array([cur_pos[:,0]+dx_.squeeze().item(), cur_pos[:,1]+dy_.squeeze().item()])
    
    euclidian_distance = np.linalg.norm(real_pos - pred_pos)

    total_err += euclidian_distance
    len += 1
    
this_err = total_err / len
print("\t\t epoch {} error : {}\n".format(args.epoch, this_err))

# json 파일을 불러오고 기록한다
with open(summary_json, "r") as j :
    summary = json.load(j)
    mean_err = float(summary["mean_err"])
    min_err = float(summary["min_err"])
    min_epoch = int(summary["min_epoch"])
    count = int(summary["count"])
    
now = 1500 - (count * 2)
count = count + 1
mean_err += this_err
if min_err > this_err : 
    summary["min_err"] = str(this_err)
    min_epoch = now
if now == 0 : 
    mean_err = mean_err / count
summary["mean_err"] = str(mean_err)
summary["min_epoch"] = str(min_epoch)
summary["count"] = str(count)

with open(summary_json, "w", encoding="utf-8") as j :
    json.dump(summary, j, ensure_ascii=False, indent='\t')