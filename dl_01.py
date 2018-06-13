import tensorflow as tf

X=tf.placeholder(tf.float32)
w=tf.Variable(tf.random_normal([2,1]))
mul=tf.matmul(X,w)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	r=sess.run(X,feed_dict={X: [1,2,3]})
	r=sess.run(w)
	print(r)
	r=sess.run(mul,feed_dict={X: [[1,2],[3,4],[5,6]]})
	print(r)