from midi_to_features import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_x, n_y = 10, 2

X_train = get_X_train()
Y_example = [1, 0]
Y_train = [[]]
for i in range (13):
    if i == 0:
        Y_train[0] = Y_example
    else:
        Y_train.append( Y_example )

    #行列互换，使其符合tensorflow的输入格式
X_train = list(map(list, zip(*X_train)))
Y_train = list(map(list, zip(*Y_train)))
print("X_train:", X_train)
print("Y_train:", Y_train)

# get testing data
X_test = get_X_test()
X_test = list(map(list, zip(*X_test)))
Y_test = [[1],[0]]
print("X_test:", X_test)
print("Y_test:", Y_test)


# Create placeholders
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y

# Initialize the parameters
def initialize_parameters():

    W1 = tf.get_variable("W1", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

    return parameters


# Forward propagation
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

# Compute Cost
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost


# the model
def model(X_train, Y_train, X_test, Y_test,learning_rate = 0.0001, num_epochs = 1000,print_cost = True):

    n_x, n_y,a = 10, 2, 13
    costs = []
    # 创建placeholder
    X, Y = create_placeholders(n_x, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 神经网络前向传播
    Z3 = forward_propagation(X, parameters)
    # 计算损失函数
    cost = compute_cost(Z3, Y)
    #用adam算法最小化损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # 创建tensorflow的session
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            _, Cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            # 打印出cost的变化
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, Cost))
            costs.append(Cost)

        # 用matplotlib绘制 时间-损失图像
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)




