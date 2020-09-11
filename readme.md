# 基于强化学习算法的推荐
本项目基于ddpg的思想，来对推荐数据生成召回，目前提供基础的demo，主要有如下几部分构成：
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


关键难点：
- 仿真环境的构建，这对于推荐非常重要，目前淘宝的一些仿真技术细节还没有时间研究
- 奖励函数的设计，目前只是简单的考虑购买价格和是否点击
- 点击序列中新增的商品id，如何进行处理，如何维护全局的商品embedding矩阵和带来潜在更新整个强化学习系统

总结： 
在我看来，使用强化学习来处理推荐系统中的相关问题，还远不成熟，而且无监督的方式比之有监督不易控制，学习过程比较困难，
不易评估，上线风险也非常大。