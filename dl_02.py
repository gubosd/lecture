'''
MNIST
no hidden layer
softmax, cross entropy, SGD, minimax
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

X=tf.placeholder(tf.float32, [None,784])
y=tf.placeholder(tf.float32, [None,10])

W=tf.Variable(tf.random_normal([784,10],stddev=0.1))
b=tf.Variable(tf.random_normal([10],stddev=0.1))

output=tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(tf.reduce_sum(-y*tf.log(output),1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op=optimizer.minimize(cost)

cost_list=[]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	batch_size=100
	total_batch=int(mnist.train.num_examples/batch_size) # 550
	
	for epoch in range(100):
		total_cost=0
		
		for i in range(total_batch):
			batch_X, batch_y = mnist.train.next_batch(batch_size)
			_, rcost = sess.run([train_op,cost],feed_dict={X: batch_X, y: batch_y})
			total_cost+=rcost
			
		cost_list.append(total_cost/total_batch)
		print('Epoch: %03d, Avg. cost = %f' % (epoch,total_cost/total_batch))
		
	pred_y=sess.run(tf.argmax(output,1), feed_dict={X: mnist.test.images})
	accuracy=np.mean(pred_y==np.argmax(mnist.test.labels,1))
	print('Accuracy = %.2f%%' % (accuracy*100))
	
plt.plot(cost_list)
plt.title('Cost per epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

'''
# display incorrectly predicted numbers
errors=np.where(pred_y!=np.argmax(mnist.test.labels,1))[0]
fig=plt.figure(figsize=(12,6))
fig.suptitle('Missing numbers')
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.imshow(mnist.test.images[errors[i]].reshape(28,28),cmap='gray_r',interpolation='none')
	plt.title(np.argmax(mnist.test.labels[errors[i]]))
	plt.xticks([])
	plt.yticks([])
plt.show()
'''