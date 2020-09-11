#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:skydm
@license: Apache Licence 
@file: ddpg.py
@time: 2020/09/02
@contact: wzwei1636@163.com
@software: PyCharm 
"""

import os
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


class DDPG:
    def __init__(self, history_length=20, embedding_size=64, hidden_dim=64, recall_length=6, batch_size=64,
                 epochs=300, gamma=0.99, critic_lr=0.002, actor_lr=0.001, tau=0.005):
        self.history_length = history_length
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.recall_length = recall_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = tau

        product_embeddings_dict, self.word2id_dict, self.id2word_dict = self.read_embedding()
        product_embeddings = np.array(list(product_embeddings_dict.values()), dtype=float)
        print(product_embeddings.shape)
        zeros_vec = np.zeros(shape=(1, self.embedding_size), dtype=float)
        self.product_embeddings_matrix = np.concatenate([zeros_vec, product_embeddings])
        self.product_embeddings = keras.layers.Embedding(len(self.word2id_dict) + 1,
                                                         self.embedding_size,
                                                         embeddings_initializer=tf.keras.initializers.Constant(
                                                             self.product_embeddings_matrix),
                                                         trainable=False)

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.df = self.read_train()
        print(self.df.head())

    def read_train(self):
        df_train = pd.read_csv('./data/train.csv', sep=";")
        # df_test = pd.read_csv('./data/test.csv', sep=";")
        for col in ['state', 'action', 'n_state', 'reward']:
            df_train[col] = df_train[col].apply(lambda x: x.split(" "))
            # df_test[col] = df_test[col].apply(lambda x: x.split(" "))
        for col in ['state', 'action', 'n_state']:
            df_train[col] = df_train[col].apply(lambda row: [self.word2id_dict[i] for i in row])
        return df_train

    def read_embedding(self):
        item_embeddings = {}
        i = 0
        with codecs.open('./data/session_word2vec.vector', 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                i += 1
                if i != 1:
                    # print(int(line.split(" ")[0]))
                    item_embeddings[str(line.split(" ")[0])] = [float(i) for i in line.split(" ")[1:]]
                    assert len(line.split(" ")[1:]) == 64
                else:
                    print("目前读取词汇大小为： %d * %d" % (int(line.split(" ")[0]), int(line.split(" ")[1])))

        # 构造词汇表
        word2id_dict = dict(zip(list(item_embeddings.keys()), range(1, len(item_embeddings.keys()) + 1)))
        id2word_dict = {id: word for word, id in word2id_dict.items()}
        return item_embeddings, word2id_dict, id2word_dict

    def get_actor(self):
        state = keras.layers.Input(shape=(self.history_length,))
        state_emb = self.product_embeddings(state)
        # output_ = keras.layers.GRU(self.hidden_dim, activation='sigmoid')(state_emb)
        output_ = keras.layers.GlobalAveragePooling1D()(state_emb)
        fc = keras.layers.Dense(self.recall_length * self.embedding_size, activation='sigmoid')(output_)
        # (batch_size, recall_length, embedding_size)
        action_weights = keras.layers.Reshape((-1, self.recall_length, self.embedding_size))(fc)
        model = keras.Model(state, action_weights)
        return model

    def get_action_item_recall(self, actions_weights):
        action_list = []
        for i in range(actions_weights.shape[0]):
            scores = np.dot(actions_weights[i], self.product_embeddings_matrix.T)
            idx = np.argmax(scores, axis=1)
            action_list.append(idx)
        return np.array(action_list)

    def save(self, path):
        actor_path = os.path.join(path, 'actor')
        critic_path = os.path.join(path, 'critic')
        self.target_actor.save_weights(actor_path + '_actor.h5')
        self.target_critic.save_weights(critic_path + '_critic.h5')

    def load_weights(self, path):
        actor_path = os.path.join(path, 'actor')
        critic_path = os.path.join(path, 'critic')
        self.target_actor.load_weights(actor_path)
        self.target_critic.load_weights(critic_path)

    def get_critic(self):
        """隐含顺序"""
        state = keras.layers.Input(shape=(self.history_length,))
        state_emb = self.product_embeddings(state)
        action = keras.layers.Input(shape=(self.recall_length,))
        action_emb = self.product_embeddings(action)
        action_emb_d = tf.cast(action_emb, dtype=tf.float32)
        action_weights = keras.layers.Input(shape=(self.recall_length, self.embedding_size), dtype=tf.float32)
        action_op = keras.layers.Multiply()([action_emb_d, action_weights])
        # output_s = keras.layers.GRU(self.hidden_dim, activation='sigmoid')(state_emb)
        output_s = keras.layers.GlobalAveragePooling1D()(state_emb)
        output_a = keras.layers.GlobalAveragePooling1D()(action_op)
        print("*" * 50)
        print(output_s.shape, output_a.shape)
        print("*" * 50)
        # output_a = keras.layers.GRU(self.hidden_dim, activation='sigmoid')(action_op)
        inputs = keras.layers.Concatenate()([output_s, output_a])
        layer1 = keras.layers.Dense(32, activation=tf.nn.relu)(inputs)
        layer2 = keras.layers.Dense(16, activation=tf.nn.relu)(layer1)
        q_value = keras.layers.Dense(1)(layer2)
        model = keras.Model([state, action, action_weights], q_value)
        print(model.summary())
        return model

    def update_target(self):
        """更新网络"""
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor.set_weights(new_weights)

    def train(self):
        # 读取batch数据
        n_batch = len(self.df) // self.batch_size

        name = "ddpg_{}_{}".format(self.epochs, self.batch_size)
        summary_writer = tf.summary.create_file_writer("logdir/static/{}".format(name))

        for i in range(self.epochs):
            for j in range(n_batch):
                batch_state = keras.preprocessing.sequence.pad_sequences(
                    self.df[j*self.batch_size:((j+1) * self.batch_size)]['state'].tolist(), maxlen=self.history_length, padding='pre', dtype='int32')

                batch_action = keras.preprocessing.sequence.pad_sequences(
                    self.df[j*self.batch_size:((j+1) * self.batch_size)]['action'].tolist(), maxlen=self.recall_length, dtype='int32')

                batch_next_state = keras.preprocessing.sequence.pad_sequences(
                    self.df[j*self.batch_size:((j+1) * self.batch_size)]['n_state'].tolist(), maxlen=self.history_length, padding='pre', dtype='int32')

                batch_reward = np.array(self.df[j*self.batch_size:((j+1) * self.batch_size)]['reward'].tolist(), dtype=np.float)

                with tf.GradientTape() as tape:
                    target_actions_weights = self.target_actor(batch_next_state)
                    target_actions_weights = tf.squeeze(target_actions_weights, 1)
                    target_actions = self.get_action_item_recall(target_actions_weights)
                    # print(len(target_actions), len(target_actions[0]))
                    # print(batch_next_state.shape, target_actions_weights.shape)
                    # print(type(batch_next_state), type(target_actions), type(target_actions_weights))

                    q_value_true = batch_reward + self.gamma * self.target_critic(
                        [batch_next_state, target_actions, target_actions_weights])
                    batch_actions_weights = self.target_actor(batch_state)  # 这一步的可以加深action权重的学习
                    batch_actions_weights = tf.squeeze(batch_actions_weights, 1)
                    # print(batch_state.shape, batch_action.shape, target_actions_weights.shape)
                    q_value_appr = self.critic_model([batch_state, batch_action, batch_actions_weights])
                    critic_loss = tf.reduce_mean(tf.square(q_value_true - q_value_appr))

                critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                keras.optimizers.Adam(self.critic_lr).apply_gradients(
                    zip(critic_grad, self.critic_model.trainable_variables))

                with tf.GradientTape() as tape:
                    actions_weights = self.actor_model(batch_state)
                    actions_weights = tf.squeeze(actions_weights, 1)
                    actions = self.get_action_item_recall(actions_weights)
                    critic_value = self.critic_model([batch_state, actions, actions_weights])
                    actor_loss = -tf.math.reduce_mean(critic_value)

                actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                keras.optimizers.Adam(self.actor_lr).apply_gradients(
                    zip(actor_grad, self.actor_model.trainable_variables))

                if j % 8 == 0:
                    print("epoch={}, actor_loss={}, critic_loss={}".format(i, actor_loss, critic_loss))
                    self.save('./model/')

                with summary_writer.as_default():
                    tf.summary.scalar("critic_loss", critic_loss, step=(i+1)*j)
                    tf.summary.scalar("actor_loss", actor_loss, step=(i+1)*j)
                    summary_writer.flush()


dp = DDPG()
dp.train()
