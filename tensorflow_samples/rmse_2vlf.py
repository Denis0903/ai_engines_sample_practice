import tensorflow as tf
import numpy as np
import matplotlib as mpl

# Declare Versions
print(np.__version__)
print(tf.__version__)
print(mpl.__version__)

# Declare Functions

def init_weight_variable(shape):
    """Initialize variable in a suitable way for weights."""
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.compat.v1.Variable(initial)


def init_bias_variable(shape):
    """Initialize variable in a suitable way for biases."""
    initial = tf.compat.v1.constant(0.1, shape=shape)
    return tf.compat.v1.Variable(initial)

with tf.compat.v1.Session() as sess:

	# Step 1. Bulid a graph

    # Preparing Datas
    x_data = np.random.random((10, 2))
    w_answer = np.array([
        [1., 2.],
        [3., 4.]
    ])
    b_answer = np.array([
        -1.,
        5.
    ])
    y_data = np.array([w_answer @ _x_data + b_answer for _x_data in x_data])
    # y_data = (w_answer @ x_data[:, :, None] + b_answer[:, None])[:, :, 0]
    # y_data = np.einsum('ij,kj->ki', w_answer, x_data) + b_answer

    # Preparing Models
    x = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, 2))
    y_answer = tf.compat.v1.placeholder(tf.compat.v1.float32)

    w = init_weight_variable((2, 2))
    b = init_bias_variable((2,))
    y_model = tf.compat.v1.matmul(x, w, transpose_b=True) + b

    # Defining Error Between Answer and Models
    loss = tf.compat.v1.compat.v1.sqrt(tf.compat.v1.reduce_mean((y_model - y_answer)**2))

    # Defining Training Argorithm
    train = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # Declare Initializer
    init = tf.compat.v1.global_variables_initializer()
    
    # Step 2. Running operation

    # Initilize
    sess.run(init)
    # Training model
    for i in range(20000):
        sess.run(train, {x: x_data, y_answer: y_data})
        if i % 1000 == 0:
            current_loss, current_w, current_b = sess.run(
                [loss, w, b], {x: x_data, y_answer: y_data})
            print(f"Loss: {current_loss}")
            print(f"w, b: \n{current_w}, {current_b}")
            # print(f"y_model: {current_y_model}, y_answer: {y_data}")