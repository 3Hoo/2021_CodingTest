import argparse

def update_parm(opt_list, loss_list) : 
    for opt in opt_list:
        opt.zero_grad()
    for loss in loss_list :
        loss.backward()
    for opt in opt_list:
        opt.step()
        
def str2bool(v) : 
    if isinstance(v, bool) :
        return v
    
    if v.lower() in ('yes', 'true', 't', 'y', '1') : 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0') : 
        return False 
    else : 
        raise argparse.ArgumentTypeError('Boolean value expected.')

