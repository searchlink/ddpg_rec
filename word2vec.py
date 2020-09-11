#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:skydm
@license: Apache Licence 
@file: word2vec.py 
@time: 2020/08/28
@contact: wzwei1636@163.com
@software: PyCharm 
"""

import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    # 1.format: 指定输出的格式和内容，format可以输出很多有用信息，
    # %(asctime)s: 打印日志的时间
    # %(levelname)s: 打印日志级别名称
    # %(message)s: 打印日志信息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    # 打印这是一个通知日志
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    # inp:分好词的文本
    # outp1:训练好的模型
    # outp2:得到的词向量
    inp, outp1, outp2 = sys.argv[1:4]

    '''
        LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
        size：是每个词的向量维度； 
        window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
        min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
        workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
        sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
        alpha (float, optional) – 初始学习率
        iter (int, optional) – 迭代次数，默认为5
    '''

    model = Word2Vec(LineSentence(inp), size=64, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save(outp1)
    # 不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(outp2, binary=False)


# python word2vec.py ./data/session_sentense.txt ./data/session_word2vec.model ./data/session_word2vec.vector