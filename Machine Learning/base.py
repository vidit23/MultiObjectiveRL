import tensorflow as tf

'''
input > weight > hidden layer 1 (activation function) > weights >
hidden layer 2 > weights > output layer

compare output to intended output > cost function (cost entropy)
optimization function (optimizer) > minimzw cost (AdamOptimizer---- SGD, AdaGrad)

backpropogation

feed foward + backprop = epoch
'''

x1 = tf.constant([5])
x2 = tf.constant([6])

result = tf.mul(x1,x2)
print(result)

with tf.Session() as sess:
	print(sess.run(result))
