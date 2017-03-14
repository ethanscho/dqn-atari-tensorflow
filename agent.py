import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import datetime as dt

env = gym.make('Breakout-v0')
observation = env.reset()

MAX_EPISODE = 10000
CROPPED_IMAGE_SIZE = 84
IMAGE_SEQUENCE_SIZE = 4
MEMORY_SIZE = 10000
BATCH_SIZE = 20
OBSERVE = 2000
GAMMA = 0.95

graph = tf.Graph()
image_sequence = list()

def forward_pass (img):
	img = tf.reshape(img, shape=[-1, 84, 84, 4])

	# Calcuate hidden layer
	h_conv1 = tf.nn.relu(tf.nn.conv2d(img, W_conv1, [1, 4, 4, 1], padding = "VALID") + b_conv1)
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, [1, 2, 2, 1], "VALID") + b_conv2)
	h_conv2_flat = tf.reshape(h_conv2,[-1, 2592])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

	out = tf.matmul(h_fc1,W_fc2) + b_fc2
	#q = tf.nn.sigmoid(out)
	return out

def process_image (img):
	gray_scaled_image = tf.image.rgb_to_grayscale(img)
	resized_image = tf.image.resize_images(gray_scaled_image, [110, 84])
	cropped_image = tf.image.resize_image_with_crop_or_pad(resized_image, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE)
	return cropped_image

def print_image (img):
	sess = tf.Session()
	with sess.as_default():
		img = img.eval()
		squeezed_img = img.squeeze();
		plt.imshow(squeezed_img)
		plt.show()

def get_action_matrix(val):
	actions = np.zeros(env.action_space.n)
	actions[val] = 1
	return actions

observation_image = tf.placeholder(tf.float32, shape=(210, 160, 3))
processed_image = process_image(observation_image)

# Create Q-Network
image_input = tf.placeholder(tf.float32, shape=(None, 4, 84, 84, 1))

W_conv1 = tf.Variable(tf.truncated_normal([8,8,4,16], stddev = 0.01))
b_conv1 = tf.Variable(tf.constant(0.01, shape = [16]))

W_conv2 = tf.Variable(tf.truncated_normal([4,4,16,32], stddev = 0.01))
b_conv2 = tf.Variable(tf.constant(0.01, shape = [32]))
#h_conv2_shape = h_conv2.get_shape().as_list()
#print "dimension:",h_conv2_shape[1]*h_conv2_shape[2]*h_conv2_shape[3]
W_fc1 = tf.Variable(tf.truncated_normal([2592, 256], stddev = 0.01))
b_fc1 = tf.Variable(tf.constant(0.01, shape = [256]))

W_fc2 = tf.Variable(tf.truncated_normal([256, env.action_space.n], stddev = 0.01))
b_fc2 = tf.Variable(tf.constant(0.01, shape = [env.action_space.n]))

#packed_images = tf.stack([image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])
q = forward_pass(image_input)
action_index = tf.argmax(q, 1)

# Training
actionInput = tf.placeholder("float", [None, env.action_space.n])
yInput = tf.placeholder("float", [None])
Q_Action = tf.reduce_sum(tf.multiply(q, actionInput), reduction_indices = 1)
cost = tf.reduce_mean(tf.square(yInput - Q_Action))
trainStep = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)

memory = list()

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)

	ckpt = tf.train.get_checkpoint_state('./tmp/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, './tmp/model.ckpt')
            print('Successfully loaded network weights')
        else:
        	print('Could not find old network weights')

	t = 0

	for episode in range (0, MAX_EPISODE):
		observation = env.reset()

		reward_sum = 0

		while True:
		    #env.render()
		    p_image = sess.run(processed_image, feed_dict={observation_image: observation})

		    image_sequence.append(p_image)

		    # Default action
		    action = 0

		    if len(image_sequence) <= IMAGE_SEQUENCE_SIZE:
		    	next_observation, reward, done, info = env.step(action)

		    else:
		    	image_sequence.pop(0)
		    	current_state = np.stack([image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])

		    	# Get action
		    	e = 0.1
		    	if np.random.rand(1) < e:
		    		action = env.action_space.sample()
		    	else:
		    		action, _q = sess.run([action_index, q], feed_dict={image_input: [current_state]})

				next_observation, reward, done, info = env.step(action) # take a random action

			    # Store in experience relay
				p_image = sess.run(processed_image, feed_dict={observation_image: next_observation})
				next_state = np.stack([image_sequence[1], image_sequence[2], image_sequence[3], p_image])
				
				#print(get_action_matrix(action))
				action_state = get_action_matrix(action)
				memory.append((current_state, action_state, reward, next_state, done)) 

				if len(memory) > MEMORY_SIZE:
					memory.pop(0)

			# Training
			if t > OBSERVE:
				# n1 = dt.datetime.now()

				minibatch = random.sample(memory, BATCH_SIZE)

				state_batch = [data[0] for data in minibatch]
				action_batch = [data[1] for data in minibatch]
				reward_batch = [data[2] for data in minibatch]
				nextState_batch = [data[3] for data in minibatch]
				terminal_batch = [data[4] for data in minibatch]

				# Step 2: calculate y 
				y_batch = []
				QValue_batch = sess.run(q, feed_dict={image_input: nextState_batch})

				for i in range(0, BATCH_SIZE):
					terminal = minibatch[i][4]
					if terminal:
						y_batch.append(reward_batch[i])
					else:
						y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

				sess.run(trainStep, feed_dict={yInput: y_batch, actionInput: action_batch, image_input: state_batch})

				# n2 = dt.datetime.now()
				# print((n2.microsecond - n1.microsecond) / 1e6)

			if t % 1000 == 0:
				saver.save(sess, './tmp/model.ckpt')

			reward_sum += reward
		    t += 1
		    observation = next_observation

		    if done:
				episode += 1
				print('Episode {} Rewards: {}'.format(episode, reward_sum))
				break;




