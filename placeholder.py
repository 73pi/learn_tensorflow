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

#定义placeholder作为存放输入数据的地方
#这里的维度也不一定要定义，但是如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(3,2), name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer() 
sess.run(init_op)

#下一行将出错：InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]
#print (sess.run(y))

#输出：[[3.957578 ]
# [1.1537654]
# [3.1674924]]
print (sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1,0.4], [0.5,0.8]]}))
sess.close()


