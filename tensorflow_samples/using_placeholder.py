import tensorflow as tf
import numpy as np
import matplotlib as mpl

# Declare Versions
print(np.__version__)
print(tf.__version__)
print(mpl.__version__)

with tf.compat.v1.Session() as sess:
	# Step 1. Bulid a graph
    v1 = tf.compat.v1.placeholder(tf.int32)
    v2 = tf.compat.v1.placeholder(tf.int32)
    add = v1 + v2

	# Step 2. Running operation

    print(sess.run(add, {v1: 3, v2: 5}))

