import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json
from itertools import chain





def seq_padding(seq, length_enc, long_length, pad_id):
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1:]
    else:
        enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq
    return enc_in

class SingleDomainSeqDataset(data.Dataset):
    def __init__(self,seq_len,view_len,isTrain,neg_nums,long_length,pad_id,subdomain,csv_path=''):
        super(SingleDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        self.user_item_data = self.user_item_data.loc[self.user_item_data['domain_id']==subdomain]

        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        if subdomain==0:
            self.seq = self.user_item_data['seq_d1'].tolist()
            self.tail_len = self.__return_tailed_length__(self.user_item_data['seq_d1'])
        elif subdomain==1:
            self.seq = self.user_item_data['seq_d2'].tolist()
            self.tail_len = self.__return_tailed_length__(self.user_item_data['seq_d2'])
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool = self.__build_i_set__(self.seq)
        print("domain 1 len:{}".format(len(self.item_pool)))
        self.seq_len = seq_len
        self.view_len = view_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
        self.subdomain = subdomain
    
    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(json.loads(x)) if len(json.loads(x)) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_tmp = json.loads(self.seq[idx])
        if len(seq_tmp)==1:
            cs_label_tmp = 1
        else:
            cs_label_tmp = 0
        if len(seq_tmp)>self.tail_len:
            tailed_label_tmp = 0
        else:
            tailed_label_tmp = 1
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        label = list()
        neg_items_set = self.item_pool - set(seq_tmp)
        item = seq_tmp[-1]
        seq_tmp = seq_tmp[:-1]
        label.append(1)
        # print("item :{}".format(item))
        # print("seq before:{}".format(seq_d1_tmp))
        while(item in seq_tmp):
            seq_tmp.remove(item)
        # print("seq after:{}".format(seq_d1_tmp))
        if self.isTrain:
            neg_samples = random.sample(neg_items_set, 1)
            label.append(0)
        else:
            neg_samples = random.sample(neg_items_set, self.neg_nums)
            for _ in range(self.neg_nums):
                label.append(0)
        seq_tmp = seq_padding(seq_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq'] = np.array([seq_tmp])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        sample['cs_label'] = np.array([cs_label_tmp])
        sample['tailed_label'] = np.array([tailed_label_tmp]) # if 1 long-tailed user
        sample['label'] = sample['label']
        return sample

def collate_fn_enhance_SD(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq = torch.cat([ torch.Tensor(sample['seq']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    cs_label = torch.cat([ torch.Tensor(sample['cs_label']) for sample in batch],dim=0)
    tailed_label = torch.cat([ torch.Tensor(sample['tailed_label']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq' : seq,
            'label':label,
            'overlap_label' : overlap_label,
            'cs_label' : cs_label,
            'tailed_label' : tailed_label,
            'neg_samples':neg_samples
            }
    return data



class DualDomainSeqDataset(data.Dataset):
    def __init__(self,seq_len,view_len,isTrain,neg_nums,long_length,pad_id,csv_path=''):
        super(DualDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        self.user_nodes = self.user_item_data['user_id'].tolist()
        # print(self.item_pool_d2)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.tail_len_d1 = self.__return_tailed_length__(self.user_item_data['seq_d1'])
        self.tail_len_d2 = self.__return_tailed_length__(self.user_item_data['seq_d2'])
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.view_len = view_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id

    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(json.loads(x)) if len(json.loads(x)) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        # print(type(seq_d1_tmp),seq_d1_tmp)
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        domain_id_old = self.domain_id[idx]
        label = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            if len(seq_d1_tmp)==1:
                cs_label_tmp = 1
            else:
                cs_label_tmp = 0
            if len(seq_d1_tmp)>self.tail_len_d1:
                tailed_label_tmp = 0
            else:
                tailed_label_tmp = 1
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d1_tmp))
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            # print("seq after:{}".format(seq_d1_tmp))
            if self.isTrain:
                neg_samples = random.sample(list(neg_items_set), 1)
                label.append(0)
            else:
                neg_samples = random.sample(list(neg_items_set), self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            if len(seq_d2_tmp)==1:
                cs_label_tmp = 1
            else:
                cs_label_tmp = 0
            if len(seq_d2_tmp)>self.tail_len_d2:
                tailed_label_tmp = 0
            else:
                tailed_label_tmp = 1
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d2_tmp))
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            # print("seq after:{}".format(seq_d2_tmp))
            if self.isTrain:
                neg_samples = random.sample(list(neg_items_set), 1)
                label.append(0)
            else:
                neg_samples = random.sample(list(neg_items_set), self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
        seq_d1_tmp = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['cs_label'] = np.array([cs_label_tmp])
        sample['tailed_label'] = np.array([tailed_label_tmp]) # if 1 long-tailed user
        sample['domain_id'] = np.array([domain_id])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        # print(neg_samples,type(neg_samples))
        sample['neg_samples'] = np.array(neg_samples)
        sample['label'] = sample['label']
        return sample

def collate_fn_enhance(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    cs_label = torch.cat([ torch.Tensor(sample['cs_label']) for sample in batch],dim=0)
    tailed_label = torch.cat([ torch.Tensor(sample['tailed_label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'cs_label':cs_label,
            'tailed_label' : tailed_label,
            'label':label,
            'domain_id' : domain_id,
            'overlap_label' : overlap_label,
            'neg_samples':neg_samples
            }
    return data

def generate_corr_seq(real_seq,fake_seq):
    seq = list()
    for i in range(len(real_seq)):
        seq.append(real_seq[i])
        seq.append(fake_seq[i])
    return seq


class DualDomainSeqDatasetC2DSR(data.Dataset):
    def __init__(self,seq_len,view_len,isTrain,neg_nums,long_length,pad_id, csv_path=''):
        super(DualDomainSeqDatasetC2DSR, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.corr_d1 = self.user_item_data['corr_d1'].tolist()
        self.corr_d2 = self.user_item_data['corr_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        self.tail_len_d1 = self.__return_tailed_length__(self.user_item_data['seq_d1'])
        self.tail_len_d2 = self.__return_tailed_length__(self.user_item_data['seq_d2'])
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.view_len = view_len
        self.pad_id = pad_id

    
    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(json.loads(x)) if len(json.loads(x)) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        # print(self.corr_d1)
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        corr_seq_1 = json.loads(self.corr_d1[idx])
        corr_seq_2 = json.loads(self.corr_d2[idx])
        domain_id_old = self.domain_id[idx]
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        label = list()
        # seq_corr = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            if len(seq_d1_tmp)==1:
                cs_label_tmp = 1
            else:
                cs_label_tmp = 0
            if len(seq_d1_tmp)>self.tail_len_d1:
                tailed_label_tmp = 0
            else:
                tailed_label_tmp = 1
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d1_tmp))
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            # print("seq after:{}".format(seq_d1_tmp))
            if self.isTrain:
                neg_samples = random.sample(list(neg_items_set), 1)
                label.append(0)
            else:
                neg_samples = random.sample(list(neg_items_set), self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
            # corr_seq = random.sample(neg_items_set, self.seq_len)

            
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            if len(seq_d2_tmp)==1:
                cs_label_tmp = 1
            else:
                cs_label_tmp = 0
            if len(seq_d2_tmp)>self.tail_len_d2:
                tailed_label_tmp = 0
            else:
                tailed_label_tmp = 1
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d2_tmp))
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            # print("seq after:{}".format(seq_d2_tmp))
            if self.isTrain:
                neg_samples = random.sample(list(neg_items_set), 1)
                label.append(0)
            else:
                neg_samples = random.sample(list(neg_items_set), self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
            # corr_seq = random.sample(neg_items_set, self.seq_len)
        
        len_d1 = min(len(seq_d1_tmp), self.seq_len)
        len_d2 = min(len(seq_d2_tmp),self.seq_len)
        seq_d1_tmp = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        corr_seq_d1 = generate_corr_seq(seq_d1_tmp,corr_seq_1)
        corr_seq_d2 = generate_corr_seq(seq_d2_tmp,corr_seq_2)
        all_seq = generate_corr_seq(seq_d1_tmp,seq_d2_tmp)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['corr_seq_d1'] = np.array([corr_seq_d1])
        sample['corr_seq_d2'] = np.array([corr_seq_d2])
        sample['all_seq'] = np.array([all_seq])
        sample['domain_id'] = np.array([domain_id])
        sample['overlap_label'] = np.array([overlap_label])
        sample['cs_label'] = np.array([cs_label_tmp])
        sample['tailed_label'] = np.array([tailed_label_tmp]) # if 1 long-tailed user
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        sample['len_d1'] = np.array([len_d1])
        sample['len_d2'] = np.array([len_d2])
        sample['label'] = sample['label']
        return sample

def collate_fn_enhanceC2DSR(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    corr_seq_d1 = torch.cat([ torch.Tensor(sample['corr_seq_d1']) for sample in batch],dim=0)
    corr_seq_d2 = torch.cat([ torch.Tensor(sample['corr_seq_d2']) for sample in batch],dim=0)
    all_seq = torch.cat([ torch.Tensor(sample['all_seq']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    cs_label = torch.cat([ torch.Tensor(sample['cs_label']) for sample in batch],dim=0)
    tailed_label = torch.cat([ torch.Tensor(sample['tailed_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    len_d1 = torch.cat([torch.Tensor(sample['len_d1']) for sample in batch], dim = 0)
    len_d2 = torch.cat([torch.Tensor(sample['len_d2']) for sample in batch], dim = 0)

    
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'corr_seq_d1' : corr_seq_d1,
            'corr_seq_d2': corr_seq_d2,
            'all_seq': all_seq,
            'overlap_label' : overlap_label,
            'tailed_label' : tailed_label,
            'cs_label':cs_label,
            'label':label,
            'domain_id' : domain_id,
            'neg_samples':neg_samples,
            'len_d1': len_d1,
            'len_d2': len_d2
            }
    return data

if __name__ == '__main__':
    datasetTrain = DualDomainSeqDatasetC2DSR(seq_len=20,view_len=100,isTrain=True,neg_nums=99,long_length=7,pad_id=111111,csv_path="/ossfs/workspace/VAE_CDR/dataset/cloth_sport_train_updated.csv")
    trainLoader = data.DataLoader(datasetTrain, batch_size=8, shuffle=False, num_workers=0,collate_fn=collate_fn_enhanceC2DSR)
    for i,sample in enumerate(trainLoader):
        # print('sample', sample['corr_seq_d1'])
        u_node = torch.LongTensor(sample['user_node'].long()).cuda()
        i_node = torch.LongTensor(sample['i_node'].long()).cuda()
        seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
        seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
        label = torch.LongTensor(sample['label'].long()).cuda()
        domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
        corr_seq_d1 = torch.LongTensor(sample['corr_seq_d1'].long()).cuda()
        corr_seq_d2 = torch.LongTensor(sample['corr_seq_d2'].long()).cuda()
        all_seq = torch.LongTensor(sample['all_seq'].long()).cuda()
        # ob_label = torch.LongTensor(sample['ob_label'].long()).cuda()
        neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
        print("u_node shape :{}".format(u_node.shape))
        print("i_node shape :{}".format(i_node.shape))
        print("seq_d1 shape :{}".format(seq_d1.shape))
        print("seq_d2 shape :{}".format(seq_d2.shape))
        print("corr_seq_d1 shape :{}".format(corr_seq_d1.shape))
        print("corr_seq_d2 shape :{}".format(corr_seq_d2.shape))
        print("all_seq shape :{}".format(all_seq.shape))
        # print("long_tail_mask_d1 shape :{}".format(long_tail_mask_d1.shape))
        # print("long_tail_mask_d2 shape :{}".format(long_tail_mask_d2.shape))
        print("label shape :{}".format(label.shape))
        print("label:".format(label))
        print("domain_id shape :{}".format(domain_id.shape))
        # print("ob_label shape :{}".format(ob_label.shape))
        print("neg_samples shape :{}".format(neg_samples.shape))
        break
