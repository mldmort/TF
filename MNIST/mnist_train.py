from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# ======================================
# looking into the data
#print (mnist.train.images[0])
#print (mnist.train.labels[0])

#print (type(mnist.train.images[0]))
print (mnist.train.images[0].size)
print (mnist.train.labels[0].size)
# ======================================
#exit()

# define placeholders (dtype, shape, name)
x = tf.placeholder(dtype=tf.float32, shape = [None, 784], name = 'ph_x')
y_ = tf.placeholder(tf.float32, [None, 10], 'ph_y')

# define Variables (initial_value, ...)
W = tf.Variable(tf.zeros([784,10]), name='var_w')
b = tf.Variable(tf.zeros([10]), name='var_b')

# model outcome
y = tf.nn.softmax(tf.matmul(x,W) + b)

# loss function, computing cross-entropy myself (probably unstable)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y) , reduction_indices = [1]))

# a more stable choice
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=tf.matmul(x,W) + b)

# training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# session decleration
sess = tf.InteractiveSession()

# initialize variables
tf.global_variables_initializer().run()

# train
for _ in range(1000):
  #print(' > iter:',_)
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# check the global accuracy of the model
check_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
precent_correct = tf.reduce_mean(tf.cast(check_prediction,tf.float32))
print('accuracy:' ,sess.run(precent_correct,{x: mnist.test.images, y_: mnist.test.labels}), '%')

# test your model on a given picture
ktest = 985
testImage = mnist.test.images[ktest]
print('y:',sess.run(y,feed_dict={x: [mnist.test.images[ktest]]}))
