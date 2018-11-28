import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2]) # None = n
Y = tf.placeholder(tf.float32, shape=[None, 1]) # None = n
w = tf.Variable(tf.random_normal([2, 1]), name='weight') # 2 = same to X, 1 = Y
b = tf.Variable(tf.random_normal([1]), name='bias') # 1 = same to Y

# Hypothesis using sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, w) + b)

#cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
  # Initialize TensorFlow variables
  sess.run(tf.global_variables_initializer())
  for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
      print(step, cost_val)

  #Accuracy report
  h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
  print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


'''
0 3.4210672
200 0.5175913
400 0.49718592
600 0.47862315
800 0.4613701
1000 0.44515145
1200 0.42981434
1400 0.4152666
1600 0.40144655
1800 0.38830855
2000 0.37581494
2200 0.36393234
2400 0.35262978
2600 0.3418777
2800 0.33164778
3000 0.32191262
3200 0.31264588
3400 0.30382195
3600 0.29541662
3800 0.28740662
4000 0.27976972
4200 0.27248496
4400 0.26553228
4600 0.25889295
4800 0.25254926
5000 0.2464844
5200 0.24068259
5400 0.23512925
5600 0.22981036
5800 0.22471295
6000 0.2198248
6200 0.2151345
6400 0.21063133
6600 0.20630507
6800 0.20214647
7000 0.19814666
7200 0.19429727
7400 0.19059052
7600 0.18701918
7800 0.18357639
8000 0.18025565
8200 0.177051
8400 0.17395675
8600 0.17096747
8800 0.1680782
9000 0.16528432
9200 0.16258116
9400 0.15996474
9600 0.15743099
9800 0.15497617
10000 0.1525968

Hypothesis:  [[0.03205536]
 [0.1605907 ]
 [0.31110126]
 [0.778549  ]
 [0.9377757 ]
 [0.9795671 ]] 
Correct (Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1.0
'''