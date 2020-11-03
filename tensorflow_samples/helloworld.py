import tensorflow as tf
import numpy as np
import matplotlib as mpl

# Declare Versions
print(np.__version__)
print(tf.__version__)
print(mpl.__version__)

with tf.compat.v1.Session() as sess:
	# Step 1. Bulid a graph
	a = tf.constant(6.0)
	b = tf.constant(5.0)
	c = a*b

	# Step 2. Running operation
	print(sess.run(c))

