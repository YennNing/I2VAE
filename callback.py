import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
import json
import sys
import os
import pickle
import random
# sys.path.append('../')
from LightGCN.code import dataloader, world, utils, register, Procedure

def load_Recmodel(csv_path):
    if "cloth" in csv_path:
        domain = 'cloth_sport'
    elif "phone" in csv_path:
        domain = 'phone_elec'
    elif "mybank" in csv_path:
        domain = 'mybank'

    dataset = dataloader.Loader(path="./LightGCN/data/"+ domain)
    n_users = dataset.n_users
    Recmodel = register.MODELS['lgn'](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    weight_file = './LightGCN/code/checkpoints/lgn-'+domain + '-3-64.pth.tar'
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    world.cprint(f"loaded model weights from {weight_file}")

    return Recmodel, n_users

def generate_fake_seq_single(user_node, real_seq, seq_len, item_set, Recmodel, n_users):
    if user_node < n_users:
        corr_seq = Procedure.Aug_Seq(user_node, real_seq, Recmodel, seq_len, world.config['multicore'])
    else:
        neg_items_set = item_set - set(real_seq)
        corr_seq = random.sample(neg_items_set, seq_len)
    return corr_seq

def build_item_set(seq1):
    item  = list()
    for item_seq in seq1:
        item_seq_list = json.loads(item_seq)
        for i_tmp in item_seq_list:
            item .append(i_tmp)
    item_pool  = set(item )
    return item_pool 

def load_hash_table(args,csv_path):
    if "cloth" in csv_path:
        domain = 'cloth_sport'
    elif "phone" in csv_path:
        domain = 'phone_elec'
    elif "mybank" in csv_path:
        domain = 'mybank'
    with open('./LightGCN/data/{}/hash_table_d2.pkl'.format(domain), 'rb') as f:
        hash_table_d2 = pickle.load(f)
    hash_table_rev = {hash_table_d2[key] : key for key in hash_table_d2}
    return hash_table_rev
    

def generate_corr_seq(args,csv_path, batch_size = 1024):
    df = pd.read_csv(csv_path)
    Recmodel, n_users = load_Recmodel(csv_path)
    all_users = df.user_id.tolist()
    # user_inf  = df[df['user_id'] < n_users_1]['user_id'].tolist()
    # user_inf_d2 = df[df['user_id'] < n_users_2]['user_id'].tolist()
    user_inf = df['user_id'].tolist()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_batches_gpu = [user_inf[i : i + batch_size] for i in range(0, len(user_inf), batch_size)]
    hash_table_d2 = load_hash_table(args, csv_path)
    item_set_d2 = list(hash_table_d2.keys())
    if 'cloth' in csv_path:
        max_item_d1 = 24506
    elif 'phone' in csv_path:
        max_item_d1 = 40714
    elif 'mybank' in csv_path:
        max_item_d1 = 64332
    print('max_item_d1:', max_item_d1)

    'Augment seqs for the domain'
    print('Augmenting ...')
    aug_seq_d1  = dict()
    aug_seq_d2 = dict()
    for batch_user in tqdm(user_batches_gpu , total = len(user_batches_gpu )):
        print('max batch user', max(batch_user))
        ratings = Recmodel.getUsersRating(torch.Tensor(batch_user).long().to('cpu'))
        filtered_df = df[df['user_id'].isin(batch_user)]
        seq1_dict = pd.Series(filtered_df['seq_d1'].values, index=filtered_df['user_id']).to_dict()
        seq2_dict = pd.Series(filtered_df['seq_d2'].values, index=filtered_df['user_id']).to_dict()
        ratings_d1 = ratings.clone()
        ratings_d1[:, max_item_d1+1: ] = -(1<<10)
        ratings_d2 = ratings.clone()
        mask = list(~np.isin(np.arange(ratings.shape[1]), list(item_set_d2)))
        ratings_d2[:, mask] =  -(1<<10)
        for idx,user_id in enumerate(batch_user ):
            seq1_items = eval(seq1_dict[user_id])
            ratings_d1[idx,seq1_items] = -(1<<10)
        _, rating_K_d1 = torch.topk(ratings_d1 , k=args.seq_len)
        for idx,user_id in enumerate(batch_user ):
            aug_seq_d1[user_id] = rating_K_d1[idx, : args.seq_len].tolist()
    
        for idx,user_id in enumerate(batch_user ):
            seq2_items = eval(seq2_dict[user_id])
            seq2_items = [iid + max_item_d1 for iid in seq2_items]
            seq2_items = [iid for iid in seq2_items if iid < ratings_d2.shape[1]]
            ratings_d2[idx,seq2_items] = -(1<<10)
        _, rating_K_d2 = torch.topk(ratings_d2 , k=args.seq_len)
        for idx,user_id in enumerate(batch_user):
            aug_seq_d2[user_id] = [hash_table_d2[iid] for iid in rating_K_d2[idx, : args.seq_len].tolist()]
    
    
    aug_seq_d1 = pd.DataFrame(list(aug_seq_d1.items()), columns = ['user_id', 'corr_d1'])
    df =  df.merge(aug_seq_d1, on = 'user_id')

    aug_seq_d2 = pd.DataFrame(list(aug_seq_d2.items()), columns = ['user_id', 'corr_d2'])
    df =  df.merge(aug_seq_d2, on = 'user_id')
    
    base_path, extension = os.path.splitext(csv_path)
    print('Write in ' + base_path + '_updated' + extension)
    df.to_csv(base_path + '_updated' + extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Callback Model')
    parser.add_argument('--seq_len', type=int, default=20, help='the length of the sequence') 
    parser.add_argument('--batch_size', type=int, default=1024, help='the length of the sequence')

    args = parser.parse_args()

    domain_names = ['cloth_sport', 'phone_elec']
    csv_paths = ['../VAE_CDR/dataset/{}_train.csv'.format(domain) for domain in domain_names]
    csv_paths += ['../VAE_CDR/dataset/{}_test_csu.csv'.format(domain) for domain in domain_names]
    for csv_path in csv_paths:
        generate_corr_seq(args, csv_path, args.batch_size)
    
    
