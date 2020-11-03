import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Declare Versions
print(np.__version__)
print(tf.__version__)


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

    # Preparing Datas_Defining Sin Func Data
    x_data = np.linspace(0., 2 * np.pi, 100)[:, None]
    y_data = np.sin(x_data)

    # Preparing Models
    x = tf.compat.v1.placeholder(tf.compat.v1.float32, (None, 1))
    y_answer = tf.compat.v1.placeholder(tf.compat.v1.float32)
    n_var = 100

    # Layer1
    w1 = init_weight_variable((1, n_var))
    b1 = init_bias_variable((n_var,))
    h1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w1) + b1)

    # Layer2
    w2 = init_weight_variable((n_var, n_var))
    b2 = init_bias_variable((n_var,))
    h2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(h1, w2) + b2)
    
    # Defining the model
    w3 = init_weight_variable((n_var, 1))
    b3 = init_bias_variable((1,))
    y_model = tf.compat.v1.matmul(h2, w3) + b3

    # Defining Error Between Answer and Models
    loss = tf.compat.v1.reduce_mean((y_model - y_answer)**2)

    # Defining Training Argorithm
    train = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # Declare Initializer
    init = tf.compat.v1.global_variables_initializer()
    
    # Step 2. Running operation

    # Initilize
    sess.run(init)
    # Training model
    for i in range(10000):
        sess.run(train, {x: x_data, y_answer: y_data})
        if i % 1000 == 0:
            current_loss, current_y_model = sess.run(
                [loss, y_model], {x: x_data, y_answer: y_data})
            print(f"Loss: {current_loss}")

    current_loss, current_y_model = sess.run(
        [loss, y_model], {x: x_data, y_answer: y_data})

    plt.plot(x_data, y_data, '.-', label='Answer')
    plt.plot(x_data, current_y_model, '.', label='Model')
    plt.legend()
    plt.show()