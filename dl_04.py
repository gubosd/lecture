'''
MNIST
CNN
conv/pool - conv/pool - hidden - output
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./mnist/data/',one_hot=True)

'''
print('mnist train data =', mnist.train.images.shape) # (55000,784)
print('mnist train labels =', mnist.train.labels.shape) # (55000,10)
print('mnist test data =', mnist.test.images.shape) # (10000,784)
print('mnist validation data =', mnist.validation.images.shape) # (5000,784)
'''

X=tf.placeholder(tf.float32, [None,28,28,1])
y=tf.placeholder(tf.float32, [None,10])

W1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
b1=tf.Variable(tf.random_normal([32],stddev=0.01))
conv_1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')+b1 # [N,28,28,32]
conv_1_relu=tf.nn.relu(conv_1)
pool_1=tf.nn.max_pool(conv_1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # [N,14,14,32]

W2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
b2=tf.Variable(tf.random_normal([64],stddev=0.01))
conv_2=tf.nn.conv2d(pool_1,W2,strides=[1,1,1,1],padding='SAME')+b2 # [N,14,14,64]
conv_2_relu=tf.nn.relu(conv_2)
pool_2=tf.nn.max_pool(conv_2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # [N,7,7,64]

W3=tf.Variable(tf.random_normal([7*7*64,256],stddev=0.1))
b3=tf.Variable(tf.random_normal([256],stddev=0.1))
hidden_1=tf.nn.relu(tf.matmul(tf.reshape(pool_2,[-1,7*7*64]),W3)+b3)

W4=tf.Variable(tf.random_normal([256,10],stddev=0.1))
b4=tf.Variable(tf.random_normal([10],stddev=0.1))
output=tf.nn.softmax(tf.matmul(hidden_1,W4)+b4)

cost=tf.reduce_mean(tf.reduce_sum(-y*tf.log(output),1))

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
train_op=optimizer.minimize(cost)

cost_list=[]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	batch_size=100
	total_batch=int(mnist.train.num_examples/batch_size) # 550
	
	for epoch in range(10):
		total_cost=0
		
		for i in range(total_batch):
			batch_X, batch_y = mnist.train.next_batch(batch_size)
			_, rcost = sess.run([train_op,cost],feed_dict={X: batch_X.reshape(-1,28,28,1), y: batch_y})
			total_cost+=rcost
			print('### %d-%d %f' % (epoch,i,rcost))
			
		cost_list.append(total_cost/total_batch)
		print('Epoch: %03d, Avg. cost = %f' % (epoch,total_cost/total_batch))
		
	pred_y=sess.run(tf.argmax(output,1), feed_dict={X: mnist.test.images.reshape(-1,28,28,1)})
	accuracy=np.mean(pred_y==np.argmax(mnist.test.labels,1))
	print('Accuracy = %.2f%%' % (accuracy*100))

'''
# Epoch: 009, Avg. cost = 0.008241
# Accuracy = 99.15%

# cost_list
[0.26349139610474759,
 0.060066648079082373,
 0.040152242099866273,
 0.030062449230304496,
 0.022707773680079053,
 0.017138848179707896,
 0.013540809828396463,
 0.011449726234740493,
 0.011226309404228231,
 0.0082405395040934144]
'''