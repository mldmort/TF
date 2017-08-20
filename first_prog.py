import tensorflow as tf

hello = tf.constant("Hello world! :) ")

node1 = tf.constant(12.0)
node2 = tf.constant(6.0)
node3 = tf.add(node1,node2)

print('hello:',hello)
print('node1:',node1)
print('node2:',node2)
print('node3:',node3)


#print("eval:",tf.Tensor.eval(hello)
exit()

# session types:
# tf.Session()
# tf.InterfactiveSession() : The only difference with a regular Session is that an InteractiveSession installs itself as the default session on construction. The methods tf.Tensor.eval and tf.Operation.run will use that session to run ops.
# So for objects which has eval() and run() defined for them, they need a session as an input but with an interactive session you don't need to pass that session as an input. (tf.constant has an eval() but doesn't have run(), but tf.global_variables_initializer() has a run())
sess = tf.Session()
#print(sess.run(node3))

a = tf.placeholder(tf.float64)
b = tf.placeholder(tf.float64)

adder_node = a + b # + provides a shortcut for tf.add(a,b)

print('a:',sess.run(a, {a: [0, 2, 4]}))
print('b:',sess.run(b, {b: [-1, -1, -1]}))
print('adder_node:',sess.run(adder_node,{a: [0, 2, 4], b: [-1, -1, -1]}))

# define placeholders for input variables
x = tf.placeholder(dtype=tf.float64)
y = tf.placeholder(tf.float64)

# define model variables which can be assigned durring the run, with their initial value
w = tf.Variable([1.],dtype=tf.float64)
b = tf.Variable([1.],dtype=tf.float64)
# must define a global initializer for variables
init = tf.global_variables_initializer()
# should run the init so that variables are initialized
sess.run(init)

# define a linear model
model = w*x + b
loss = tf.reduce_sum(tf.square(model - y))

# cheating build an exact y_ex
w_ex = tf.constant([-1.0],dtype=tf.float64)
b_ex = tf.constant(-5.0,dtype=tf.float64)
print('w_ex:',sess.run(w_ex))
print('b_ex:',sess.run(b_ex))
y_ex = w_ex*x+b_ex

print('x:',sess.run(x,{x:[-1., 2., 0., 1., 5.]}))
print('y_ex:',sess.run(y_ex,{x:[-1., 2., 0., 1., 5.]}))

print("> test loss with w, b")
print('w:',sess.run(w))
print('b:',sess.run(b))
print('loss:',sess.run(loss,{x:[-1., 2., 0., 1., 5.],y:[ -4.,  -7.,  -5.,  -6., -10.]}))

#print("> test loss with w, b")
#fix_w = tf.assign(w,[-1.])
#fix_b = tf.assign(b,[-5.])
#sess.run(fix_w)
#sess.run(fix_b)
#print('w:',sess.run(w))
#print('b:',sess.run(b))
#print('loss:',sess.run(loss,{x:[-1., 2., 0., 1., 5.],y:[ -4.,  -7.,  -5.,  -6., -10.]}))

# can we train the model by minimizing the loss with tf?
# need an optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
# train definition
train = optimizer.minimize(loss)
for i in range(300):
  sess.run(train,{x:[-1., 2., 0., 1., 5.],y:[ -4.,  -7.,  -5.,  -6., -10.]})
  #print(sess.run([w,b]))

# take a look...
print(sess.run([w, b]))
