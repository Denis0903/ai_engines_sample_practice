import tensorflow as tf
import numpy as np
import matplotlib as mpl

# Declare Versions
print(np.__version__)
print(tf.__version__)
print(mpl.__version__)

with tf.compat.v1.Session() as sess:

	# Step 1. Bulid a graph

    # Preparing Datas
    x_data = np.linspace(0., 1., 6)  # => => [0.  0.2 0.4 0.6 0.8 1. ]
    a_answer = 1.5
    b_answer = .1
    y_data = a_answer * x_data + b_answer  # => [0.1 0.4 0.7 1.  1.3 1.6]

    # Preparing Models
    x = tf.compat.v1.placeholder(tf.float32)
    y_answer = tf.compat.v1.placeholder(tf.float32)
    a_model = tf.compat.v1.Variable(1.0)
    b_model = tf.compat.v1.Variable(0.0)
    y_model = a_model * x + b_model

    # Defining Error Between Answer and Models
    loss = tf.compat.v1.sqrt(tf.compat.v1.reduce_mean((y_model - y_answer)**2))

    # Defining Training Argorithm
    train = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(loss)

    # Declare Initializer
    init = tf.compat.v1.global_variables_initializer()
    
    # Step 2. Running operation

    # Initilize
    sess.run(init)
    # Training model
    for i in range(20000):
        sess.run(train, {x: x_data, y_answer: y_data})
        if i % 1000 == 0:
            current_loss, current_y_model = sess.run(
                [loss, y_model], {x: x_data, y_answer: y_data})
            print(f"Loss: {current_loss}")
            print(f"y_model: {current_y_model}, y_answer: {y_data}")



