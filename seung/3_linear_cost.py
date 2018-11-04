import tensorflow as tf
x_data=[1,2,3]
y_data=[1,2,3]

#W=tf.Variable(tf.random_normal([1]), name='weight')
W=tf.Variable(5.0)
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#Our hypothesis for linear model X*W
H=X*W

#cost/loss function
cost=tf.reduce_sum(tf.square(H-Y))

#MINIMIZE:gradient descent using derivative: W-=learning_rate*derivative
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

#launch the graph in a session
sess=tf.Session()
#global variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(100):
  sess.run(update, feed_dict={X:x_data, Y:y_data})
  #W will be match to 1
  print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

