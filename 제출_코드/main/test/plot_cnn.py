'''
훈련이 완료된 모델을 load한 후,
test set을 이용해 모델의 성능을 test하는 모듈
이때 load하는 모델은 calculate.sh 를 통해 test set에 대해 가장 낮은 err을 가진 모델이다

모델이 주어진 input에 대한 posX와 posY의 변화량을 출력하면
test data의 cur pos에 그 변화량을 더하여 최종 예상 posX와 posY를 그린다.

이때 그래프에 지난 30 step의 pos도 함께 찍어서 경로를 그리고,
실제 next pos와 모델을 통해 예측한 predicted pos를 다른 색으로 그려서
모델이 올바르게 예측했는지 직관적으로 판단할 수 있도록 한다
'''

import numpy as np
import pickle
import os
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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
parser.add_argument('--model_dir', type=str)
parser.add_argument('--plot_dir', type=str)

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
    
model_dir = args.model_dir

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

# 총 315개의 test data, 20개 마다의 텀을 주어서 사진을 출력하도록 해보자
img_save_term = 20

for self_idx, (cur_pos, next_pos, x, target) in enumerate(test_loader) : 
    dx_ = model_d(x)
    dy_ = model_d(x)
    real_pos = np.array([next_pos[:,0], next_pos[:, 1]])
    pred_pos = np.array([cur_pos[:,0]+dx_.squeeze().item(), cur_pos[:,1]+dy_.squeeze().item()])
    
    euclidean_distance = np.linalg.norm(real_pos - pred_pos)
    
    if self_idx % img_save_term == 0 :
        plt.cla()
        plt.subplot(1, 1, 1)
        plt.scatter(real_pos[0,:], real_pos[1,:], c='g', s=150, label="Real Next Pos")
        plt.scatter(pred_pos[0,:], pred_pos[1,:], c='r', s=150, label="Predicted Next Pos")
        plt.plot([real_pos[0,:], pred_pos[0,:]], [real_pos[1,:], pred_pos[1,:]], c='darkviolet', label="euclidean Distance")
        txt = plt.text(x=(real_pos[0,:]+pred_pos[0,:])/2, y=(real_pos[1,:]+pred_pos[1,:])/2, s=str(euclidean_distance), c='yellow', size='large')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])
        plt.scatter(cur_pos[:,0], cur_pos[:,1], c='b', s=70, label="Current Pos")
        plt.legend()
        plt.savefig(os.path.join(args.plot_dir, str(self_idx)+'.png'))
    