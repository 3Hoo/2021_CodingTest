import torch
import torch.nn.functional as F
import numpy as np

'''
loss log를 관리하는 클래스

[ member var ]
    - log_book                  : [loss name : loss list] 쌍을 저장하는 딕셔너리
        
[ member func ]
    - alloc_stat_type           : log_book에 key를 추가
    - alloc_stat_type_list      : log_book에 input으로 받은 리스트의 원소들을 key로 추가
    - init_stat                 : log_book의 모든 key들의 value를 empty list로 초기화
    - add_stat                  : log_book의 key의 loss list에 input을 append
    - add_torch_stat            : log_book의 key의 loss list에 torch tensor의 loss를 받아와 append
    - get_stat                  : log_book의 key의 loss의 평균을 return
    - print_stat                : log_book의 key의 모든 loss 평균을 print
'''
class LogManager:
    def __init__(self):
        self.log_book=dict()

    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []

    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)

    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []

    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
        
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
        
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
           stat = self.get_stat(stat_type)
           if stat != 0:
            print(stat_type,":",stat, end=' / ')
        print(" ")
        
def l2loss(pred, target) : 
    return F.mse_loss(pred, target)

def calc_err(pred, target) : 
    err = torch.mean(torch.square(pred - target))
    return err