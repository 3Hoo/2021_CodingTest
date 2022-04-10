import numpy as np
import pickle
import os
import time
import argparse
import sys

import torch
import torch.optim as optim

import data_manager as dm
from model import lstm_dX, lstm_dY
from loss import LogManager, l2loss, calc_err
from utils import update_parm, str2bool

    
# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--base_lr', type=float)
parser.add_argument('--c_lr', type=float)
parser.add_argument('--bUseAngleV', default=False, type=str2bool)
parser.add_argument('--epochs', type=int)
parser.add_argument('--model_save_path', type=str)
parser.add_argument('--seed', type=int)

args = parser.parse_args()

assert args.model_type in ['LSTM_Attention', 'Transformer', 'CNN', 'ViT']

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Data Load
TRAIN_DATA_PATH = "../data/1222/train_data.p"
VAL_DATA_PATH = "../data/1222/val_data.p"
TEST_DATA_PATH = "../data/1222/test_data.p"

with open(TRAIN_DATA_PATH, "rb") as p :
    TRAIN_DATA = pickle.load(p)
with open(VAL_DATA_PATH, "rb") as p :
    VAL_DATA = pickle.load(p)
with open(TEST_DATA_PATH, "rb") as p :
    TEST_DATA = pickle.load(p)
    

if torch.cuda.is_available() : 
    device = "cuda"
else :
    device = "cpu"
model_dir = args.model_save_path

lr = args.base_lr
c_lr = args.c_lr

time_step = 100
if args.bUseAngleV : 
    latent_dim = 7
    hidden_dim = 10
else :
    latent_dim = 4
    hidden_dim = 6
batch_size = 10

if args.model_type == "LSTM_Attention" : 
    model_dX = lstm_dX(device=device, time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)
    model_dY = lstm_dY(device=device, time_step=time_step, latent_dim=latent_dim, hidden_dim=hidden_dim, batch_size=batch_size)


model_dX.cuda()
model_dY.cuda()
model_dX_opt = optim.Adam(model_dX.parameters(), lr = lr)
model_dY_opt = optim.Adam(model_dY.parameters(), lr = lr)
model_dX_sch = optim.lr_scheduler.LambdaLR(optimizer=model_dX_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/(args.epochs))*epoch+1e-2))
model_dY_sch = optim.lr_scheduler.LambdaLR(optimizer=model_dY_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/(args.epochs))*epoch+1e-2))
  
torch.save(model_dX.state_dict(), os.path.join(model_dir, "final_dX.pt"))
torch.save(model_dY.state_dict(), os.path.join(model_dir, "final_dY.pt"))

total_time = 0

min_dev_dX_loss = 99999999
min_dev_dY_loss = 99999999
min_dX_epoch = 0
min_dY_epoch = 0

lm = LogManager()
lm.alloc_stat_type_list(["dX_loss", "dY_loss", "dX_err", "dY_err", "total_loss"])

for epoch in range(args.epochs+1) : 
    print("EPOCH :      {}      LearningRate_dX :      {}       LeraningRate_dY :       {}".format(epoch, model_dX_sch.get_last_lr()[0], model_dY_sch.get_last_lr()[0]))
    print()
    
    
    # Train Step
    lm.init_stat()
    train_start_time = time.time()
    
    model_dX.train()
    model_dY.train()
    
    train_loader = dm.train_data_loader(TRAIN_DATA, batch_size=batch_size, device="cuda", use_angle_v=args.bUseAngleV, shuffle=False)
    
    for self_idx, (cur_pos, next_pos, x, target) in enumerate(train_loader) : 
        total_loss = 0.0

        # data load checking
        '''
        print(cur_pos.shape)
        print(next_pos.shape)
        print(x.shape)
        print(target.shape)
        print(x.get_device)
        print(target[:,0].get_device)
        print(next(model_dX.parameters()).is_cuda)
        print(next(model_dY.parameters()).is_cuda)
        '''
        
        # dX
        dx_ = model_dX(x, target[:,0])
        dx_loss = l2loss(dx_.squeeze(), target[:,0])
        dx_err = calc_err(dx_.squeeze(), target[:,0])
        total_loss += dx_loss.item()
        #print("------------------------------")
        #print("dx : {} | t : {}".format(dx_[0], target[:,0][0]))
        
        # dY
        dy_ = model_dY(x, target[:,1])
        dy_loss = l2loss(dy_.squeeze(), target[:,1])
        dy_err = calc_err(dy_.squeeze(), target[:,1])
        total_loss += dy_loss.item()
        #print("dy : {} | t : {}".format(dy_[0], target[:,1][0]))
        
        '''
        print("------------------------")
        print(type(dx_loss))
        print(type(dy_loss))
        print(type(dx_err))
        print(type(dx_err))
        print(type(total_loss))
        '''
        
        lm.add_torch_stat("dX_loss", dx_loss)
        lm.add_torch_stat("dY_loss", dy_loss)
        lm.add_torch_stat("dX_err", dx_err)
        lm.add_torch_stat("dY_err", dy_err)
        lm.add_stat("total_loss", total_loss)
        
        opt_list = [model_dX_opt, model_dY_opt]
        loss_list = [dx_loss, dy_loss]
        update_parm(opt_list, loss_list)
        
    print("TRAIN : ", end=' ')
    lm.print_stat()
    
    train_time = time.time() - train_start_time
    total_time += train_time
    
    # Eval Step
    eval_start_time = time.time()
    lm.init_stat()
    
    model_dX.eval()
    model_dY.eval()
    
    dev_loader = dm.train_data_loader(VAL_DATA, batch_size=batch_size, device="cuda", use_angle_v=args.bUseAngleV, shuffle=False)
    
    with torch.no_grad() :
        for self_idx, (cur_pos, next_pos, x, target) in enumerate(dev_loader) :
            total_loss = 0.0
        
            # dX
            dx_ = model_dX(x, target[:,0])
            dx_loss = l2loss(dx_.squeeze(), target[:,0])
            dx_err = calc_err(dx_.squeeze(), target[:,0])
            total_loss += dx_loss.item()
            
            # dY
            dy_ = model_dY(x, target[:,1])
            dy_loss = l2loss(dy_.squeeze(), target[:,1])
            dy_err = calc_err(dy_.squeeze(), target[:,1])
            total_loss += dy_loss.item()
            
            lm.add_torch_stat("dX_loss", dx_loss)
            lm.add_torch_stat("dY_loss", dy_loss)
            lm.add_torch_stat("dX_err", dx_err)
            lm.add_torch_stat("dY_err", dy_err)
            lm.add_stat("total_loss", total_loss)
        
        print("DEV: ", end=' ')
        lm.print_stat()
    
    print(".....................")
    
    model_dX_sch.step()
    model_dY_sch.step()
    
    # min loss check & model save
    cur_dX_loss = lm.get_stat("dX_loss")
    cur_dY_loss = lm.get_stat("dY_loss")
    
    if min_dev_dX_loss > cur_dX_loss :
        min_dev_dX_loss = cur_dX_loss
        min_dX_epoch = epoch
    if min_dev_dY_loss > cur_dY_loss :
        min_dev_dY_loss = cur_dY_loss
        min_dY_epoch = epoch
    
    torch.save(model_dX.state_dict(), os.path.join(model_dir, "model_dX", "parm", str(epoch)+".pt"))
    torch.save(model_dY.state_dict(), os.path.join(model_dir, "model_dY", "parm", str(epoch)+".pt"))
    
print("***********************************")
print("Model name:",model_dir.split("/")[-1])
print("TIME PER EPOCH:",total_time/args.epochs)
print("dX Final Epoch:",min_dX_epoch, min_dev_dX_loss)
print("dY Final Epoch:",min_dY_epoch, min_dev_dY_loss)
print("***********************************")

os.system("cp "+os.path.join(model_dir, "model_dX", "parm",str(min_dX_epoch)+".pt")+" "+os.path.join(model_dir,"final_dX.pt"))
os.system("cp "+os.path.join(model_dir, "model_dY", "parm",str(min_dX_epoch)+".pt")+" "+os.path.join(model_dir,"final_dY.pt"))