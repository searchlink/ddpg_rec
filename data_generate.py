#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:skydm
@license: Apache Licence 
@file: data_generate.py 
@time: 2020/09/01
@contact: wzwei1636@163.com
@software: PyCharm 
"""
import os
import csv
import codecs
import random
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataGenerator:
    def __init__(self):
        self.df = pd.read_csv("./data/nc_rl_data.txt", sep='|', names=['sess_id', 'productid', 'price', 'timestamp'])
        self.df = self.df.groupby("sess_id").filter(lambda x: len(x) > 6)
        print(self.df.shape)
        self.df['price'] = round(np.log2(self.df['price'] + 1), 4)
        self.users = self.df['sess_id'].unique()
        self.items = self.df['productid'].unique()
        if os.path.exists('./data/history.pkl'):
            with open('./data/history.pkl', 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = self.get_history()
            with open('./data/history.pkl', 'wb') as file:
                pickle.dump(self.history, file)
        print(len(self.history))
        print(self.history[:1])

    def gen_word2vec_vocab(self, file):
        with codecs.open(file, 'w') as fin:
            for user_history in self.history:
                fin.writelines(" ".join([str(i) for i in user_history['productid'].tolist()]) + '\n')
        fin.close()

    def get_history(self):
        historic_users = []  # 用户历史点击序列
        for i, u in tqdm(enumerate(self.users)):
            temp = self.df[self.df['sess_id'] == u]
            temp = temp.sort_values('timestamp').reset_index()  # 默认升序
            temp.drop('index', axis=1, inplace=True)
            historic_users.append(temp)
        return historic_users

    def get_train_test(self, train_ratio=0.8, seed=None):
        n = len(self.history)
        if seed is not None:
            random.Random(seed).shuffle(self.history)

        self.train = self.history[:int(train_ratio * n)]
        self.test = self.history[int(train_ratio * n):]

    def gen_sample_data(self, user_history, num_action=6):
        '''单个用户构造的样本集'''
        states = []
        actions = []
        next_states = []
        rewards = []

        n = len(user_history)
        # print("长度n:", n)
        # print(user_history)

        for i in range(1, n - num_action):
            states.append(user_history['productid'][:i].tolist())
            actions.append(user_history['productid'][i:(i+num_action)].tolist())
            next_states.append(user_history['productid'][:i].tolist() + user_history['productid'][i:(i+num_action)].tolist())
            rewards.append(user_history['price'][i:(i+num_action)].tolist())
        # print('states:', states)
        # print('actions:', actions)
        # print('next_states:', next_states)
        # print('rewards:', rewards)
        return states, actions, next_states, rewards

    def write_csv(self, filename, history_to_write, delimiter=';'):
        with open(filename, "w", newline='') as f:
            f_writer = csv.writer(f, delimiter=delimiter)
            f_writer.writerow(['state', 'action', 'n_state', 'reward'])
            for user_history in history_to_write:
                states, actions, n_states, rewards = self.gen_sample_data(user_history)
                for i in range(len(states)):
                    state_str = ' '.join([str(j) for j in states[i]])
                    action_str = ' '.join([str(j) for j in actions[i]])
                    n_state_str = ' '.join([str(j) for j in n_states[i]])
                    reward_str = ' '.join([str(j) for j in rewards[i]])
                    f_writer.writerow([state_str, action_str, n_state_str, reward_str])


dg = DataGenerator()
# dg.get_train_test(train_ratio=0.8, seed=1234)
# dg.write_csv('./data/train.csv', dg.train)
# dg.write_csv('./data/test.csv', dg.test)
dg.gen_word2vec_vocab('./data/session_sentense.txt')