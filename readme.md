# 基于强化学习算法的推荐
本项目基于ddpg的思想，来对推荐数据生成召回，主要有如下几部分构成：
- 数据获取(由于涉及隐私信息，这里提供了部分样例数据)     
样例数据位于data/nc_rl_data_sample.txt中
- 构造生成('state', 'action', 'n_state', 'reward')格式的数据和gensim调用word2vec指定
的数据格式      
代码位于data_generate.py中
- 预训练生成商品的embedding矩阵和词汇表   
代码位于word2vec.py中
- 模型及训练           
代码位于ddpg.py

训练了过程如下： 
![actor损失函数](https://github.com/searchlink/ddpg_rec/blob/master/image/actor.png)
![critic损失函数](https://github.com/searchlink/ddpg_rec/blob/master/image/critic.png)