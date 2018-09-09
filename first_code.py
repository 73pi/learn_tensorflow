#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : first_code.py
# @Author: wxc
# @Date  : 2018/9/8

import tensorflow as tf
#取消额外的输出（...compiled to use: AVX2）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([3.0,4.0],name='b')

result01 = a+b
result02 = a*b

sess = tf.Session()
sess.run(result01)
print sess.run(result01)
print sess.run(result02)


