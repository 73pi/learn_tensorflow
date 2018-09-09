#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : first_code.py
# @Author: wxc
# @Date  : 2018/9/9

import tensorflow as tf
#取消额外的输出（...compiled to use: AVX2）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#声明w1，w2两个变量
#这里还通过seed参数设定了随机种子,这能保证每次运行结果都是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

#暂时将输入的特征向量定义为一个常量,注意这里x是一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])

#前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
#这里与之前的代码不同，不能直接通过sess.run(y)来获取y的取值
#因为w1，w2都还没有运行初始化过程
#初始化w1，w2
sess.run(w1.initializer)
sess.run(w2.initializer)

#输出[[3.957578]]
print (sess.run(y))
sess.close()


