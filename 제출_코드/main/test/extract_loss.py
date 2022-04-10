'''
train log 파일에서 loss 값을 추출하기 위한 간단한 모듈
'''

import os

log_file = "./extracted_loss_log/1224_trainLog_DSA_30step_to2000.txt"
save_path = "./extracted_loss_log/DSA"

train_dx_losses = []
train_dy_losses = []
train_total_losses = []
dev_dx_losses = []
dev_dy_losses = []
dev_total_losses = []

with open(log_file, "r") as log :
    for l in log.readlines() :
        type = l.split(" :  ", maxsplit=1)[0]
        
        if type == "TRAIN" : 
            tmp = l.split(" :  ", maxsplit=1)[1]
            train_dx_loss = tmp.split(" / ")[0].split(" : ")[-1]
            train_dy_loss = tmp.split(" / ")[1].split(" : ")[-1]
            train_total_loss = tmp.split(" / ")[-2].split(" : ")[-1]
            train_dx_losses.append(train_dx_loss)
            train_dy_losses.append(train_dy_loss)
            train_total_losses.append(train_total_loss)
        elif type == "DEV" : 
            tmp = l.split(" :  ", maxsplit=1)[1]
            dev_dx_loss = tmp.split(" / ")[0].split(" : ")[-1]
            dev_dy_loss = tmp.split(" / ")[1].split(" : ")[-1]
            dev_total_loss = tmp.split(" / ")[-2].split(" : ")[-1]
            dev_dx_losses.append(dev_dx_loss)
            dev_dy_losses.append(dev_dy_loss)
            dev_total_losses.append(dev_total_loss)
        else :
            continue
        
with open(os.path.join(save_path, "train_loss.txt"), "w") as f :
    f.write("dx loss...\n")
    for i in range(len(train_dx_losses)) : 
        f.write(str(train_dx_losses[i]+'\n'))
    f.write("---------------\n\n")
    
    f.write("dy loss...\n")
    for i in range(len(train_dy_losses)) : 
        f.write(str(train_dy_losses[i]+'\n'))
    f.write("---------------\n\n")
    
    f.write("total loss...\n")
    for i in range(len(train_total_losses)) : 
        f.write(str(train_total_losses[i]+'\n'))
    f.write("---------------\n\n")
    
    
with open(os.path.join(save_path, "dev_loss.txt"), "w") as f :
    f.write("dx loss...\n")
    for i in range(len(dev_dx_losses)) : 
        f.write(str(dev_dx_losses[i]+'\n'))
    f.write("---------------\n\n")
    
    f.write("dy loss...\n")
    for i in range(len(dev_dy_losses)) : 
        f.write(str(dev_dy_losses[i]+'\n'))
    f.write("---------------\n\n")
    
    f.write("total loss...\n")
    for i in range(len(dev_total_losses)) : 
        f.write(str(dev_total_losses[i]+'\n'))
    f.write("---------------\n\n")