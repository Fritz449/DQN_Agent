import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.placeholder('float32')
b = tf.placeholder('float32')
c = a+b
sess.run(tf.initialize_all_variables())

print c.eval(feed_dict = {a:2,b:3})