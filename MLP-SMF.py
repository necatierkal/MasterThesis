#!/usr/bin/env python
# coding: utf-8

# The MIT License (MIT)
# 
# Copyright (c) 2021 NVIDIA CORPORATION
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# In[ ]:


import argparse
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#fname = 'cpmp_092' mlp_cpmp_092
fname = 'mlp_cpmp_092'
checkpoint_path = Path('./checkpoints') / fname

# if not checkpoint_path.exists():
#     checkpoint_path.mkdir(parents=True)
# else:
#     sys.exit()
    
input_path = Path('01_Data/')


# In[ ]:


LOW_CITY_THR = 9


# In[ ]:


import logging
import os
import random
import time
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
import wandb
import cudf
import cupy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import gc


# In[ ]:


def shift_feature(df, groupby_col, col, offset, nan=-1, colname=''):
    df[colname] = df[col].shift(offset)
    df.loc[df[groupby_col]!=df[groupby_col].shift(offset), colname] = nan


# In[ ]:


pd.options.display.max_columns = 100


# In[ ]:


def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# In[ ]:


# Return top4 metric
# istest: flag to select if metric should be computed in 0:train, 1:test,
# pos: select which city to calculate the metric, 0: last, 1: last-1, 2:last-2 , -1: all
# the input `val` dataframe must contains the target `city_id` and the 4 recommendations as: rec0, res1, rec2 and rec3

def top4_metric( val, istest=0, pos=0 , target='city_id'):
    
    if istest>=0:
        val = val.loc[ (val.submission==0) & (val.istest == istest) ]
    else:
        val = val.loc[ (val.submission==0) ]

    if pos >= 0:
        top1 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec0']
        top2 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec1']
        top3 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec2']
        top4 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec3']
    else:
        top1 = val[target] == val['rec0']
        top2 = val[target] == val['rec1']
        top3 = val[target] == val['rec2']
        top4 = val[target] == val['rec3']
        
    return (top1|top2|top3|top4).mean()    


# In[ ]:


raw = cudf.read_csv(input_path / 'train_and_test_2.csv')
#raw = cudf.read_csv(input_path / 'train_and_test_2_azaltilmisYeni.csv')
#raw = cudf.read_csv(input_path / 'train_and_test_2_NextCity_Eksiltilmis_Dominionsuz.csv')
#raw = cudf.read_csv(input_path / 'train_and_test_2_NextCity_Eksiltilmis.csv')
############################################################# Cartesian product##################################
#raw = cudf.read_csv(input_path / 'train_and_test_2_NextCity.csv')   
################################################################################################################
# raw = cudf.read_csv(input_path / 'train_and_test_2_Deneme.csv') #nrows=5
# raw = cudf.read_csv(input_path / 'train_and_test_2_Deneme_2.csv') #nrows=5
# raw = cudf.read_csv(input_path / 'train_and_test_2_Deneme_3.csv') #nrows=5

print(raw.shape)


# In[ ]:


raw.loc[raw['city_id'] == 0, 'city_id'] = np.NaN


# In[ ]:


df = raw[(raw.istest == 0) | (raw.icount > 0)].groupby('city_id').utrip_id.count().reset_index()
df

df.columns = ['city_id', 'city_count']
raw = raw.merge(df, how='left', on='city_id')
raw.loc[raw.city_count <= LOW_CITY_THR, 'city_id'] = -1
raw = raw.sort_values(['utrip_id', 'checkin'])


# In[ ]:


CATS = ['city_id', 'hotel_country', 'booker_country', 'device_class']
MAPS = []
for c in CATS:
    raw[c+'_'], mp = raw[c].factorize()
    MAPS.append(mp)
    print('created', c+'_')   


# In[ ]:


LOW_CITY = np.where(MAPS[0].to_pandas() == -1)[0][0]
LOW_CITY


# In[ ]:


NUM_CITIES = raw.city_id_.max()+1
NUM_HOTELS = raw.hotel_country_.max()+1
NUM_DEVICE = raw.device_class_.max() + 1


# In[ ]:


raw['reverse'] = 0
rev_raw = raw[raw.istest == 0].copy()
rev_raw['reverse'] = 1
rev_raw['utrip_id'] = rev_raw['utrip_id']+'_r'


# In[ ]:


tmp = rev_raw['icount'].values.copy()
rev_raw['icount'] = rev_raw['dcount']
rev_raw['dcount'] = tmp
rev_raw = rev_raw.sort_values(['utrip_id', 'dcount']).reset_index(drop=True)
raw = cudf.concat([raw, rev_raw]).reset_index(drop=True)


# In[ ]:


raw['sorting'] = cupy.asarray(range(raw.shape[0]))


# In[ ]:


raw['utrip_id'+'_'], mp = raw['utrip_id'].factorize()


# In[ ]:


# ENGINEER LAG FEATURES
LAGS=5
lag_cities = []
lag_countries = []

for i in range(1,LAGS+1):
    shift_feature(raw, 'utrip_id_', 'city_id_', i, NUM_CITIES, f'city_id_lag{i}')
    lag_cities.append(f'city_id_lag{i}')
    shift_feature(raw, 'utrip_id_', 'hotel_country_', i, NUM_HOTELS, f'country_lag{i}')
    lag_countries.append(f'country_lag{i}')


# In[ ]:


lag_cities


# In[ ]:


#lag_countries = lag_countries[:1]
lag_countries


# In[ ]:


tmpD = raw[raw['dcount']==0][['utrip_id', 'city_id_']]
tmpD.columns = ['utrip_id', 'first_city']
raw = raw.merge(tmpD,on='utrip_id',how='left')
tmpD = raw[raw['dcount']==0][['utrip_id', 'hotel_country_']]
tmpD.columns = ['utrip_id', 'first_country']
raw = raw.merge(tmpD,on='utrip_id',how='left')


# In[ ]:

raw['checkin'] = cudf.to_datetime(raw.checkin, format="%Y-%m-%d")
raw['checkout'] = cudf.to_datetime(raw.checkout, format="%Y-%m-%d")


# In[ ]:


raw['mn'] = raw.checkin.dt.month
raw['dy1'] = raw.checkin.dt.weekday
raw['dy2'] = raw.checkout.dt.weekday
raw['length'] = cupy.log1p((raw.checkout - raw.checkin).dt.days) 



# In[ ]:


tmpD = raw[raw['dcount']==0][['utrip_id', 'checkin']]
tmpD.columns = ['utrip_id', 'first_checkin']
raw = raw.merge(tmpD,on='utrip_id',how='left')
tmpD = raw[raw['icount']==0][['utrip_id', 'checkout']]
tmpD.columns = ['utrip_id', 'last_checkout']
raw = raw.merge(tmpD,on='utrip_id',how='left')


# In[ ]:


raw['trip_length'] = ((raw.last_checkout - raw.first_checkin).dt.days)
raw['trip_length'] = cupy.log1p(cupy.abs(raw['trip_length'])) * cupy.sign(raw['trip_length'])


# In[ ]:


tmpD = raw[raw['icount']==0][['utrip_id', 'checkin']]
tmpD.columns = ['utrip_id', 'last_checkin']
raw = raw.merge(tmpD,on='utrip_id',how='left')
tmpD = raw[raw['dcount']==0][['utrip_id', 'checkout']]
tmpD.columns = ['utrip_id', 'first_checkout']
raw = raw.merge(tmpD,on='utrip_id',how='left')
raw['trip_length'] = raw['trip_length'] - raw['trip_length'].mean()


# In[ ]:


raw = raw.sort_values('sorting')


# In[ ]:


shift_feature(raw, 'utrip_id_', 'checkout', 1, None, f'checkout_lag{1}')


# In[ ]:


raw['lapse'] = (raw['checkin'] - raw['checkout_lag1'] ).dt.days.fillna(-1)


# In[ ]:


# ENGINEER WEEKEND AND SEASON
raw['day_name']= raw.checkin.dt.weekday
raw['weekend']=raw['day_name'].isin([5,6]).astype('int8')
df_season = cudf.DataFrame({'mn': range(1,13), 'season': ([0]*3)+([1]*3)+([2]*3)+([3]*3)})
raw=raw.merge(df_season, how='left', on='mn')
raw = raw.sort_values(['sorting'], ascending=True)


# In[ ]:


raw.head(10)


# In[ ]:


_ = plt.hist(raw['lapse'].to_pandas(), bins=100, log=True)
raw['lapse'].mean(), raw['lapse'].std()


# In[ ]:


_ = plt.hist(raw['N'].to_pandas(), bins=100, log=True)
raw['N'].mean(), raw['N'].std()
# _=plt.hist(raw['PairCount'].to_pandas(),bins=100,log=True)
# raw['pairCount'].mean(),raw['pairCount'].std()


#raw['Possibility'] = raw.Possibility #Ben ekledim.
#raw['Possibility'] = raw.Possibility #Ben ekledim.
#raw['Possibility'] = raw['Possibility'].fillna(0).astype(int)

#raw['PairCount'] = raw.PairCount #Ben ekledim.
#raw['PairCount'] = raw['PairCount'].fillna(0).astype(int)

#raw['NextCity'] = raw.NextCity #Ben ekledim.
#raw['NextCity'] = raw['NextCity'].fillna(0).astype(int)

# In[ ]:


raw['N'] = raw['N'] - raw['N'].mean()
raw['N'] /= 3


# In[ ]:


_ = plt.hist(raw['trip_length'].to_pandas(), bins=100, log=True)
raw['trip_length'].mean(), raw['length'].std()


# In[ ]:


_ = plt.hist(raw['length'].to_pandas(), bins=100, log=True)
raw['length'].mean(), raw['length'].std()


# In[ ]:


raw['log_icount'] = cupy.log1p(raw['icount'])
raw['log_dcount'] = cupy.log1p(raw['dcount'])


# In[ ]:


_ = plt.hist(raw['log_icount'].to_pandas(), bins=100, log=True)
raw['log_icount'].mean(), raw['log_icount'].std()


# In[ ]:


_ = plt.hist(raw['log_dcount'].to_pandas(), bins=100, log=True)
raw['log_dcount'].mean(), raw['log_dcount'].std()


# In[ ]:


raw['mn'].unique()


# In[ ]:


raw['dy1'].unique()


# In[ ]:


raw['dy2'].unique()
#raw['Possibility'].unique() #Ben ekledim.


# In[ ]:


class BookingDataset(Dataset):
    def __init__(self,
                 data,
                 target=None,
                ):
        super(BookingDataset, self).__init__()
        self.lag_cities_ = data[lag_cities].values
        self.mn = data['mn'].values - 1
        self.dy1 = data['dy1'].values
        self.dy2 = data['dy2'].values
 #       self.PairCount = data['PairCount'].values #Ben ekledim.
        self.length = data['length'].values
        self.trip_length = data['trip_length'].values
        self.N = data['N'].values
 #       self.Possibility = data['Possibility'].values #Ben ekledim.
 #       self.NextCity = data['NextCity'].values #Ben ekledim.
        self.log_icount = data['log_icount'].values
        self.log_dcount = data['log_dcount'].values
        self.lag_countries_ = data[lag_countries].values
        self.first_city = data['first_city'].values
        self.first_country = data['first_country'].values
        self.booker_country_ = data['booker_country_'].values
        self.device_class_ = data['device_class_'].values
        self.lapse = data['lapse'].values
        self.season = data['season'].values
        self.weekend = data['weekend'].values
        if target is None:
            self.target = None
        else:
            self.target = data[target].values
        
    def __len__(self):
        return len(self.lag_cities_)
        
    def __getitem__(self, idx: int):
        input_dict = {
            'lag_cities_': torch.tensor(self.lag_cities_[idx], dtype=torch.long),
            'mn': torch.tensor([self.mn[idx]], dtype=torch.long),
            'dy1': torch.tensor([self.dy1[idx]], dtype=torch.long),
            'dy2': torch.tensor([self.dy2[idx]], dtype=torch.long),
  #          'Possibility' : torch.tensor([self.Possibility[idx]], dtype=torch.float), #Ben ekledim
  #          'PairCount' :  torch.tensor([self.PairCount[idx]], dtype=torch.float), #Ben ekledim
  #          'NextCity' :  torch.tensor([self.NextCity[idx]], dtype=torch.float), #Ben ekledim
            'length': torch.tensor([self.length[idx]], dtype=torch.float),
            'trip_length': torch.tensor([self.trip_length[idx]], dtype=torch.float),
            'N': torch.tensor([self.N[idx]], dtype=torch.float),
            'log_icount': torch.tensor([self.log_icount[idx]], dtype=torch.float),
            'log_dcount': torch.tensor([self.log_dcount[idx]], dtype=torch.float),
            'lag_countries_': torch.tensor(self.lag_countries_[idx], dtype=torch.long),
            'first_city': torch.tensor([self.first_city[idx]], dtype=torch.long),
            'first_country': torch.tensor([self.first_country[idx]], dtype=torch.long),
            'booker_country_': torch.tensor([self.booker_country_[idx]], dtype=torch.long),
            'device_class_': torch.tensor([self.device_class_[idx]], dtype=torch.long),
            'lapse': torch.tensor([self.lapse[idx]], dtype=torch.float),
            'season': torch.tensor([self.season[idx]], dtype=torch.long),
            'weekend': torch.tensor([self.weekend[idx]], dtype=torch.long),
        }
        if self.target is not None:
            input_dict['target'] = torch.tensor([self.target[idx]], dtype=torch.long)
        return input_dict


# In[ ]:


dataset = BookingDataset(raw.to_pandas(), 'city_id_')

dataset.__getitem__(3) #ben yoruma aldım
#dataset.__getitem__(5) #ben ekledim


# In[ ]:


def train_epoch(loader, model, optimizer, scheduler, scaler, device):

    model.train()
    model.zero_grad()
    train_loss = []
    bar = tqdm(range(len(loader)))
    load_iter = iter(loader)
    #batch = load_iter.next()
    batch = next(load_iter)
    batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
    
    for i in bar:
        
        old_batch = batch
        if i + 1 < len(loader):
            # batch = load_iter.next()
            batch = next(load_iter)
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
                    

        out_dict = model(old_batch)
        logits = out_dict['logits']
        loss = out_dict['loss']              
        loss_np = loss.detach().cpu().numpy()
        
        loss.backward()

        optimizer.step()
        scheduler.step()
        for p in model.parameters(): 
            p.grad = None

        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, model, device):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
       
    with torch.no_grad():
        bar = tqdm(range(len(loader)))
        load_iter = iter(loader)
        # batch = load_iter.next()
        batch = next(load_iter)
        batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }


        for i in bar:

            old_batch = batch
            if i + 1 < len(loader):
                # batch = load_iter.next()
                batch = next(load_iter)
                batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }

            out_dict = model(old_batch)
            logits = out_dict['logits']
            loss = out_dict['loss']              
            loss_np = loss.detach().cpu().numpy()
            target = old_batch['target']
            LOGITS.append(logits.detach())
            TARGETS.append(target.detach())
            val_loss.append(loss_np) 

            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    
    return val_loss, LOGITS, TARGETS
        


# In[ ]:


def save_checkpoint(model, optimizer, scheduler, scaler, best_score, fold, seed, fname):
    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_score': best_score,
        }
    torch.save(checkpoint, './checkpoints/%s/%s_%d_%d.pt' % (fname, fname, fold, seed))


# In[ ]:


def load_checkpoint(fold, seed, device, fname):
    model = Net(NUM_CITIES+1, NUM_HOTELS+1, EMBEDDING_DIM, HIDDEN_DIM, dropout_rate=DROPOUT_RATE,
                loss=False).to(device)
 
    checkpoint = torch.load('./checkpoints/%s/%s_%d_%d.pt' % (fname, fname, fold, seed))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


# In[ ]:


loss_fct = torch.nn.CrossEntropyLoss(ignore_index=LOW_CITY)

class Net(nn.Module):

    def __init__(self, num_cities, num_countries, embedding_dim, hidden_dim, dropout_rate, loss=True):
        super(Net, self).__init__()
        self.loss = loss
        self.dropout_rate = dropout_rate
        
        self.cities_embeddings = nn.Embedding(num_cities, embedding_dim)     
        # self.cities_embeddings.weight.data.normal_(0., 0.01)
        self.cities_embeddings.weight.data.normal_(0., 0.01)
        print('city embedding data shape', self.cities_embeddings.weight.shape)

        self.countries_embeddings = nn.Embedding(num_countries, embedding_dim)     
        # self.countries_embeddings.weight.data.normal_(0., 0.01)
        self.countries_embeddings.weight.data.normal_(0., 0.01)
        print('country embedding data shape', self.countries_embeddings.weight.shape)
        ###################################################################
        #self.NextCity_embedding = nn.Embedding(num_cities, embedding_dim)   #Ben ekledim NextCity_embedding  
        #self.NextCity_embedding.weight.data.normal_(0., 0.01)
        #print('nextcity embedding data shape', self.NextCity_embedding.weight.shape)
        ###################################################################
    
        
        self.mn_embeddings = nn.Embedding(12, embedding_dim)     
        # self.mn_embeddings.weight.data.normal_(0., 0.01)
        self.mn_embeddings.weight.data.normal_(0., 0.01)

        self.dy1_embeddings = nn.Embedding(7, embedding_dim)     
        # self.dy1_embeddings.weight.data.normal_(0., 0.01)
        self.dy1_embeddings.weight.data.normal_(0., 0.01)

        self.dy2_embeddings = nn.Embedding(7, embedding_dim)     
        # self.dy2_embeddings.weight.data.normal_(0., 0.01)
        self.dy2_embeddings.weight.data.normal_(0., 0.01)
        
        ###################################################################
       # self.PairCount_embeddings = nn.Embedding(1, embedding_dim)   #Ben ekledim PairCount embedding 
       # self.PairCount_embeddings.weight.data.normal_(0., 0.01)
        ###################################################################
        
        ###################################################################
        #self.Possibility_embeddings = nn.Embedding(1, embedding_dim)   #Ben ekledim Possibility_embeddings  
        #self.Possibility_embeddings.weight.data.normal_(0., 0.01)
        ###################################################################        
        
        # self.season_embeddings = nn.Embedding(7, embedding_dim)     
        # self.season_embeddings.weight.data.normal_(0., 0.01)
        
        # self.season_embeddings = nn.Embedding(4, embedding_dim)     
        # # self.season_embeddings.weight.data.normal_(0., 0.01)
        # self.season_embeddings.weight.data.normal_(0., 0.01)
        
        self.weekend_embeddings = nn.Embedding(2, embedding_dim)     
        # self.weekend_embeddings.weight.data.normal_(0., 0.01)
        self.weekend_embeddings.weight.data.normal_(0., 0.01)
        
        self.linear_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_length = nn.BatchNorm1d(embedding_dim)
        # self.norm_length = nn.InstanceNorm1d(embedding_dim)
        self.activate_length = nn.ReLU()
        #self.activate_length = nn.Sigmoid()
        
        self.linear_trip_length = nn.Linear(1, embedding_dim, bias=False)
        self.norm_trip_length = nn.BatchNorm1d(embedding_dim)
        # self.norm_trip_length = nn.InstanceNorm1d(embedding_dim)
        self.activate_trip_length = nn.ReLU()
        #self.activate_trip_length = nn.Sigmoid()

        self.linear_N = nn.Linear(1, embedding_dim, bias=False)
        self.norm_N = nn.BatchNorm1d(embedding_dim)
        # self.norm_N = nn.InstanceNorm1d(embedding_dim)
        self.activate_N = nn.ReLU()
        #self.activate_N = nn.Sigmoid()
        
       # self.linear_Possibility= nn.Linear(1,embedding_dim,bias=False)
       # self.norm_Possibility= nn.BatchNorm1d(embedding_dim)
       # self.activate_Possibility= nn.ReLU()

        
        self.linear_log_icount = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_icount = nn.BatchNorm1d(embedding_dim)
        # self.norm_log_icount = nn.InstanceNorm1d(embedding_dim)
        self.activate_log_icount = nn.ReLU()
        #self.activate_log_icount = nn.Sigmoid()

        self.linear_log_dcount = nn.Linear(1, embedding_dim, bias=False)
        self.norm_log_dcount = nn.BatchNorm1d(embedding_dim)
        # self.norm_log_dcount = nn.InstanceNorm1d(embedding_dim)
        self.activate_log_dcount = nn.ReLU()
        #self.activate_log_dcount = nn.Sigmoid()

        self.devices_embeddings = nn.Embedding(NUM_DEVICE, embedding_dim)     
        # self.devices_embeddings.weight.data.normal_(0., 0.01)
        self.devices_embeddings.weight.data.normal_(0., 0.01)
        print('device_embeddings data shape', self.devices_embeddings.weight.shape)

        self.linear_lapse = nn.Linear(1, embedding_dim, bias=False)
        self.norm_lapse = nn.BatchNorm1d(embedding_dim)
        # self.norm_lapse = nn.InstanceNorm1d(embedding_dim)
        self.activate_lapse = nn.ReLU()
        #self.activate_lapse = nn.Sigmoid()
        
        self.linear1 = nn.Linear((len(lag_cities) + len(lag_countries) + 1)*embedding_dim, hidden_dim)
        # self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        # self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.activate1 = nn.PReLU()
        #self.activate1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
                   
        ##########################################################
        # rnn_hidden_dim=64
        # rnn_num_layers=1
        # self.GRU = nn.GRU(hidden_dim,hidden_dim,num_layers=rnn_num_layers,batch_first=True)
        ##########################################################
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.activate2 = nn.PReLU()
        #self.activate2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        ############################################################
        # neco_dim=256
        self.linear21 = nn.Linear(hidden_dim,hidden_dim)
        self.norm21 = nn.BatchNorm1d(hidden_dim)
        self.activate21 = nn.PReLU()
        #self.activate21 = nn.Sigmoid()
        self.dropout21= nn.Dropout(self.dropout_rate)
        
        # self.linear22 = nn.Linear(hidden_dim,hidden_dim)
        # self.norm22 = nn.BatchNorm1d(hidden_dim)
        # self.activate22 = nn.PReLU()
        # self.dropout22= nn.Dropout(self.dropout_rate)
        # ############################################################
        self.linear3 = nn.Linear(hidden_dim, embedding_dim)
        self.norm3 = nn.BatchNorm1d(embedding_dim)
        # self.norm3 = nn.InstanceNorm1d(embedding_dim)
        self.activate3 = nn.PReLU()
        #self.activate3 = nn.Sigmoid()
        self.dropout3 = nn.Dropout(self.dropout_rate)
 
        self.output_layer_bias = nn.Parameter(torch.Tensor(num_cities, ))
        self.output_layer_bias.data.normal_(0., 0.01)
        
    def get_embed(self, x, embed):
        bs = x.shape[0]
        x = embed(x)      
        # lag_embed.shape: bs, x.shape[1], embedding_dim
        x = x.view(bs, -1)
        return x

    def forward(self, input_dict):
        lag_embed = self.get_embed(input_dict['lag_cities_'], self.cities_embeddings)      
        lag_countries_embed = self.get_embed(input_dict['lag_countries_'], self.countries_embeddings)      
        mn_embed = self.get_embed(input_dict['mn'], self.mn_embeddings)      
        dy1_embed = self.get_embed(input_dict['dy1'], self.dy1_embeddings)      
        dy2_embed = self.get_embed(input_dict['dy2'], self.dy2_embeddings)  
        # pairCount = input_dict['pairCount'].long()  # ya da .int() #Ben ekledim.
        # pairCount_embed = self.get_embed(pairCount, self.pairCount_embeddings) #Ben ekledim.

        #pairCount_embed = self.get_embed(input_dict['pairCount'],self.pairCount_embeddings) #Ben ekledim.
        # season_embed = self.get_embed(input_dict['season'], self.season_embeddings)  #Burayı açtım
        weekend_embed = self.get_embed(input_dict['weekend'], self.weekend_embeddings)  
        length = input_dict['length']
        length_embed = self.activate_length(self.norm_length(self.linear_length(length)))
        trip_length = input_dict['trip_length']
        trip_length_embed = self.activate_trip_length(self.norm_trip_length(self.linear_trip_length(trip_length)))
        N = input_dict['N']
        N_embed = self.activate_N(self.norm_N(self.linear_N(N)))
        #PairCount=input_dict['PairCount'] #Ben ekledim.
        #PairCount_embed=self.activate_Possibility(self.norm_Possibility(self.linear_Possibility(PairCount)))#Ben ekledim. Bu işlemden sonra değerler nan oluyor.
        #NextCity=input_dict['Possibility'] #Ben ekledim.
        #NextCity_embed=self.activate_Possibility(self.norm_Possibility(self.linear_Possibility(NextCity)))#Ben ekledim. Bu işlemden sonra değerler nan oluyor.
        #Possibility=input_dict['Possibility'] #Ben ekledim.
        #Possibility_embed=self.activate_Possibility(self.norm_Possibility(self.linear_Possibility(Possibility)))#Ben ekledim. Bu işlemden sonra değerler nan oluyor.
        lapse = input_dict['lapse']
        ###################################################################
        # rnn_input= torch.cat([lag_embed,lag_countries_embed],-1)
        # rnn_output = self.rnn(rnn_input) #Ben ekledim rnn denemesi
        ###################################################################
        lapse_embed = self.activate_lapse(self.norm_lapse(self.linear_lapse(lapse)))
        log_icount = input_dict['log_icount']
        log_icount_embed = self.activate_log_icount(self.norm_log_icount(self.linear_log_icount(log_icount)))
        log_dcount = input_dict['length']
        log_dcount_embed = self.activate_log_dcount(self.norm_log_dcount(self.linear_log_dcount(log_dcount)))
        first_city_embed = self.get_embed(input_dict['first_city'], self.cities_embeddings)  
        first_country_embed = self.get_embed(input_dict['first_country'], self.countries_embeddings)  
        booker_country_embed = self.get_embed(input_dict['booker_country_'], self.countries_embeddings)  
     
        device_embed = self.get_embed(input_dict['device_class_'], self.devices_embeddings)  #+ season_embed pairCount_embed x e eklendi.
        x = (mn_embed + dy1_embed + dy2_embed + length_embed + log_icount_embed +log_dcount_embed \
             + first_city_embed + first_country_embed + booker_country_embed + device_embed \
             + trip_length_embed + N_embed + lapse_embed + weekend_embed) #+ Possibility_embed + NextCity_embed+PairCount_embed)
        x = torch.cat([lag_embed, lag_countries_embed, x], -1)
        # rnn_output_tensor = rnn_output[0]
        # x = torch.cat([rnn_output_tensor], -1)  #Ben ekledim rnn denemesi
        x = self.activate1(self.norm1(self.linear1(x)))
        x = self.dropout1(x)
        
        #################################################################        
        # x = self.GRU(x)
        # x = x[0]
        #################################################################
        x = x + self.activate2(self.norm2(self.linear2(x)))
        x = self.dropout2(x)
        ############################################################
        x = self.activate21(self.norm21(self.linear21(x)))
        x = self.dropout21(x)
        # x = self.activate22(self.norm22(self.linear22(x)))
        # x = self.dropout22(x)
        ############################################################
        x = self.activate3(self.norm3(self.linear3(x)))
        x = self.dropout3(x)
   
        logits = F.linear(x, self.cities_embeddings.weight, bias=self.output_layer_bias)
        output_dict = {
            'logits':logits
                      }
        if self.loss:
            target = input_dict['target'].squeeze(1)
            #print(logits.shape, target.shape)
            loss = loss_fct(logits, target)
            output_dict['loss'] = loss
        return output_dict


# In[ ]:


TRAIN_BATCH_SIZE = 1024
WORKERS = 8
LR = 1e-3
EPOCHS = 12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# EPOCHS = 20
GRADIENT_ACCUMULATION = 1
EMBEDDING_DIM = 64
# EMBEDDING_DIM = 32
HIDDEN_DIM =  1024
# DROPOUT_RATE = 0.2
DROPOUT_RATE = 0.2
device = torch.device('cuda')
#displayName = "Linear21 added azltilmis data en az 5 şehir"
displayName = "Linear21 added orijinal data"

model = Net(NUM_CITIES+1, NUM_HOTELS+1, EMBEDDING_DIM, HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

wandb.init(
    # set the wandb project where this run will be logged
    project="mlp-smf",
    name = displayName,
    config= {
            "TRAIN_BATCH_SIZE" : TRAIN_BATCH_SIZE,
            #WORKERS = 8
            "WORKERS" : WORKERS,
            "LR" : LR,
            "EPOCHS" : EPOCHS,
            "GRADIENT_ACCUMULATION" : GRADIENT_ACCUMULATION,
            "EMBEDDING_DIM" : EMBEDDING_DIM,
            "HIDDEN_DIM" :  HIDDEN_DIM,
            "DROPOUT_RATE" : DROPOUT_RATE,
            "optimizer" : optimizer,
            "loss_fct" : loss_fct,           
            }
    # track hyperparameters and run metadata
    #config={
    #"learning_rate": 0.02,
    #"architecture": "CNN",
    #"dataset": "CIFAR-100",
    #"epochs": 10,
    #}
)
# In[ ]:


def get_top4(preds):
    TOP4 = np.empty((preds.shape[0], 4))
    for i in range(4):
        x = np.argmax(preds, axis=1)
        TOP4[:,i] = x
        x = np.expand_dims(x, axis=1)
        np.put_along_axis(preds, x, -1e10, axis=1)
    return TOP4

def top4(preds, target):
    TOP4 = get_top4(preds)
    acc = np.max(TOP4 == target, axis=1)
    acc = np.mean(acc)
    return acc


# In[ ]:


TRAIN_WITH_TEST = True

seed = 0
seed_torch(seed)

preds_all = []
best_scores = []
best_epochs = []
for fold in range(5):

    seed_torch(seed)
    preds_fold = []
    print('#'*25)
    print('### FOLD %i'%(fold))
    if TRAIN_WITH_TEST:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&(raw.istest==0)|( (raw.istest==1)&(raw.icount>0) ) ].copy()
    else:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&(raw.istest==0) ].copy()
    valid = raw.loc[ (raw.fold==fold)&(raw.istest==0)&(raw.icount==0) &(raw.reverse == 0)].copy()
    print(train.shape, valid.shape)

    train_dataset = BookingDataset(train.to_pandas(), target='city_id_')

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=True,
        pin_memory=True,
    )

    valid_dataset = BookingDataset(valid.to_pandas(), target='city_id_')

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    

    model = Net(NUM_CITIES+1, NUM_HOTELS+1, EMBEDDING_DIM, HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                              pct_start=0.1, 
                                            #   pct_start=0.01,
                                            #   div_factor=1e3, 
                                              div_factor=1e3,
                                              max_lr=3e-3, 
                                              epochs=EPOCHS, 
                                              steps_per_epoch=int(np.ceil(len(train_data_loader)/GRADIENT_ACCUMULATION)))
    scaler = GradScaler()

    best_score = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(time.ctime(), 'Epoch:', epoch, flush=True)
        train_loss = train_epoch(train_data_loader, model, optimizer, scheduler, scaler, device)
        val_loss, PREDS, TARGETS = val_epoch(valid_data_loader, model, device) 
        PREDS[:, LOW_CITY] = -1e10# remove low frequency cities
        score = top4(PREDS, TARGETS)
        wandb.log({"train_loss": np.mean(train_loss), "validation_loss" : np.mean(val_loss), "score" : score, "fold": fold, "epoch": epoch})
        content = 'Fold %d Seed %d Ep %d lr %.7f train loss %4f val loss %4f score %4f'
        print(content % (fold, seed, epoch, 
                         optimizer.param_groups[0]["lr"],
                         np.mean(train_loss),
                         np.mean(val_loss),
                         score,
                        ), 
              flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            preds_fold = PREDS
            save_checkpoint(model, optimizer, scaler, scheduler, best_score, fold, seed, fname)
    del model, scaler, scheduler, optimizer, valid_data_loader, valid_dataset, train_data_loader, train_dataset
    gc.collect()

    preds_all.append(preds_fold)
    print('fold %d, best score: %0.6f best epoch: %3d' % (fold, best_score, best_epoch))
    best_scores.append(best_score)
    best_epochs.append(best_epoch)
    #with open('../checkpoints/%s/%s_%d_preds.pkl' % (fname, fname, seed), 'wb') as file:
    #    pkl.dump(preds_all, file)
        
    #break
print()
for fold, (best_score, best_epoch) in enumerate(zip(best_scores, best_epochs)):
    print('fold %d, best score: %0.6f best epoch: %3d' % (fold, best_score, best_epoch))
print('seed %d best score: %0.6f best epoch: %0.1f' % (seed, np.mean(best_scores), np.mean(best_epochs)))


# In[ ]:


def test_epoch(loader, models, device):

    #model.eval()
    PREDS = []
    
    with torch.no_grad():
        if 1:
            bar = tqdm(range(len(loader)))
            load_iter = iter(loader)
            # batch = load_iter.next()
            batch = next(load_iter)
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }

            for i in bar:
                old_batch = batch
                if i + 1 < len(loader):
                    # batch = load_iter.next()
                    batch = next(load_iter)
                    batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
                    
                preds = 0
                for model in models:
                    out_dict = model(old_batch)
                    preds = preds + out_dict['logits'] / NFOLDS
                PREDS.append(preds.detach())
                
    
    PREDS = torch.cat(PREDS).cpu().numpy()
    
    return PREDS

NFOLDS = 5
seed = 0
models = [load_checkpoint(fold, seed, device, fname) for fold in range(NFOLDS)]


# In[ ]:


pd.options.display.max_columns = 100


# ## Test N-1 (leaky as we train using all test)

# In[ ]:


def acc(valid):
    acc = cupy.max(valid[COLS[1:]].values == valid[['city_id']].values, axis=1)
    acc = cupy.mean(acc)
    return acc


# In[ ]:


test = raw.loc[ (raw.istest==1)&(raw.icount==1) ].copy()
print( test.shape )
test.head()


# In[ ]:


test_dataset = BookingDataset(test.to_pandas(), target=None)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=WORKERS,
    shuffle=False,
    pin_memory=True,
)

PREDS = test_epoch(test_data_loader, models, device) 
PREDS[:, LOW_CITY] = -1e10# remove low frequency cities
TOP4 = get_top4(PREDS).astype('int')
TOP4.shape

COLS = ['utrip_id']


# In[ ]:


# city_mapping = MAPS[0].reset_index()
city_mapping = cudf.DataFrame(index=MAPS[0].to_pandas()).reset_index()


# In[ ]:


CITY_MAP = MAPS[0].astype('int')
for k in range(4):
    test['city_id_%i'%(k+1)] = TOP4[:,k]
    tmp = test[['city_id_%i'%(k+1)]].astype('int32').copy()
    tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
    tmp = tmp.merge(city_mapping, how='left', left_on='city_id_%i'%(k+1), right_on='index')
    tmp = tmp.sort_values('sorting')
    # test['city_id_%i'%(k+1)] = tmp['city_id'].astype('int32').values.copy()
    test['city_id_%i'%(k+1)] = tmp['city_id_%i'%(k+1)].astype('int32').values.copy()
    COLS.append('city_id_%i'%(k+1))
    
test[COLS].head()


# In[ ]:


test.head()


# In[ ]:


acc(test)


# ## Test submission

# In[ ]:


test = raw.loc[ (raw.istest==1)&(raw.icount==0)&(raw.reverse==0) ].copy()
print( test.shape )
test.head()


# In[ ]:


test_dataset = BookingDataset(test.to_pandas(), target=None)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=WORKERS,
    shuffle=False,
    pin_memory=True,
)

PREDS = test_epoch(test_data_loader, models, device) 
PREDS[:, LOW_CITY] = -1e10# remove low frequency cities
TOP4 = get_top4(PREDS).astype('int')
TOP4.shape

COLS = ['utrip_id']
CITY_MAP = MAPS[0].astype('int')
for k in range(4):
    test['city_id_%i'%(k+1)] = TOP4[:,k]
    tmp = test[['city_id_%i'%(k+1)]].astype('int32').copy()
    tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
    tmp = tmp.merge(city_mapping, how='left', left_on='city_id_%i'%(k+1), right_on='index')
    tmp = tmp.sort_values('sorting')
    # test['city_id_%i'%(k+1)] = tmp['city_id'].astype('int32').values.copy()
    test['city_id_%i'%(k+1)] = tmp['city_id_%i'%(k+1)].astype('int32').values.copy()
    COLS.append('city_id_%i'%(k+1))
    
test[COLS].head()


# In[ ]:


test[COLS].to_csv('%s_sub.csv' % fname, index=False)


# ## OOF Prediction

# In[ ]:


def load_checkpoint(fold, seed, device, fname, loss=False):
    model = Net(NUM_CITIES+1, NUM_HOTELS+1, EMBEDDING_DIM, HIDDEN_DIM, dropout_rate=DROPOUT_RATE,
                loss=loss).to(device)
 
    checkpoint = torch.load('./checkpoints/%s/%s_%d_%d.pt' % (fname, fname, fold, seed))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_topN(preds, N):
    TOPN = np.empty((preds.shape[0], N))
    PREDN = np.empty((preds.shape[0], N))
    preds = preds.copy()
    for i in tqdm(range(N)):
        x = np.argmax(preds, axis=1)
        TOPN[:,i] = x
        x = np.expand_dims(x, axis=1)
        PREDN[:,i] = np.take_along_axis(preds, x, axis=1).ravel()
        np.put_along_axis(preds, x, -1e10, axis=1)
    return TOPN, PREDN

def get_top4(preds):
    preds = preds.copy()
    TOP4 = np.empty((preds.shape[0], 4))
    for i in range(4):
        x = np.argmax(preds, axis=1)
        TOP4[:,i] = x
        x = np.expand_dims(x, axis=1)
        np.put_along_axis(preds, x, -1e10, axis=1)
    return TOP4

def top4(preds, target):
    TOP4 = get_top4(preds)
    acc = np.max(TOP4 == target, axis=1)
    acc = np.mean(acc)
    return acc

TRAIN_WITH_TEST = True

seed = 0
seed_torch(seed)

preds_all = []
test_preds_all = []
train_all = []
best_scores = []
for fold in range(1):

    seed_torch(seed)
    preds_fold = []
    print('#'*25)
    print('### FOLD %i'%(fold))
    valid = raw.loc[ (raw.fold==fold)&(raw.istest==0)&(raw.icount==0) &(raw.reverse == 0)].copy()
    print(valid.shape)

    valid_dataset = BookingDataset(valid.to_pandas(), target='city_id_')

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=True,
    )
    
    test_dataset = BookingDataset(test.to_pandas(), target=None)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    
    model = load_checkpoint(fold, seed, device, fname, loss=True)
    val_loss, PREDS, TARGETS = val_epoch(valid_data_loader, model, device) 
    PREDS[:, LOW_CITY] = -1e10# remove low frequency cities
    score = top4(PREDS, TARGETS)
    print('fold %d, best score: %0.6f' % (fold, score))
    score = top4(PREDS, valid[['city_id_']].to_pandas().values)
    print('fold %d, best score: %0.6f' % (fold, score))
    best_scores.append(score)
    preds_all.append(PREDS)
    train_all.append(valid.to_pandas())

    model = load_checkpoint(fold, seed, device, fname, loss=False)
    TEST_PREDS = test_epoch(test_data_loader, [model], device) 
    TEST_PREDS[:, LOW_CITY] = -1e10# remove low frequency cities
    test_preds_all.append(TEST_PREDS)
print('seed %d best score: %0.6f' % (seed, np.mean(best_scores)))

preds_all = np.concatenate(preds_all)
print('vaid pred shape', preds_all.shape)

test_preds_all = np.mean(test_preds_all, axis=0)
print('test pred shape', test_preds_all.shape)

top_preds, top_logits = get_topN(preds_all, 4)
targets = np.concatenate([valid[['city_id_']].values for valid in train_all])
valid_trips = np.concatenate([valid[['utrip_id']].values for valid in train_all])

print('CV score', np.mean(np.max(top_preds == targets, axis=1)))

valid_trips

top_preds, top_logits = get_topN(preds_all, 50)
top_test_preds, top_test_logits = get_topN(test_preds_all, 50)


top_preds.shape, top_logits.shape, top_test_preds.shape, top_test_logits.shape

valid_cities = np.concatenate([valid[['city_id']].values for valid in train_all])
oof = {
    'valid_trips':valid_trips,
    'top_preds':top_preds,
    'top_logits':top_logits,
    'top_test_preds':top_test_preds,
    'top_test_logits':top_test_logits,
    'city_map':CITY_MAP,
    'valid_cities':valid_cities,
}

with open((checkpoint_path / (fname + '_oof.pkl')), 'wb') as file:
    pkl.dump(oof, file)
    
print('done')


# In[ ]:


oof = {
    'valid_trips':valid_trips,
    'top_preds':top_preds,
    'top_logits':top_logits,
    'top_test_preds':top_test_preds,
    'top_test_logits':top_test_logits,
    'city_map':CITY_MAP,
    'valid_cities':valid_cities,
    'preds_all':preds_all,
    'test_preds_all':test_preds_all,
}

with open((checkpoint_path / (fname + '_oof.pkl')), 'wb') as file:
    pkl.dump(oof, file, protocol=pkl.HIGHEST_PROTOCOL)
    
print('done')

