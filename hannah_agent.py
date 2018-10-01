from frozenlakegame import frozenlakegame
import numpy as np
import tensorflow as tf
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""agent.py: Implementation of random action agent for COSC470 Assignment 3.
"""

__author__      = "Lech Szymanski"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "lechszym@cs.otago.ac.nz"

''' CONFIGURABLE PARAMETERS '''
online = True
chance_explore = 0.3
weight = 0.99
device = '/gpu:0'
n_filters_conv1 = 512
filter_size_conv1 = 2
stride1 = 1
n_filters_conv2 = 16
filter_size_conv2 = 2
stride2 = 1
#n_filters_conv3 = 32
#filter_size_conv3 = 2
#stride3 = 1
fc1_layer_size = 1024
exp = int(chance_explore*10)
w = int(weight*100)
id = str(n_filters_conv1)+"-"+str(filter_size_conv1)+"-"+ \
     str(fc1_layer_size)+"_"+str(exp)+"_"+str(w) # used to name output text files, saved models, and graphs to identify
#str(n_filters_conv3) + "-" + str(filter_size_conv3) + "-" +
# Instantiate the game

# You can change the reward value of the ice squares - by default it's 0, but it
# might be a good idea to give it a small negative reward if you want your agent
# to pick shorter paths
env = frozenlakegame(R=-0.01)

# Number of learning episodes
num_episodes = 100000 # one hundred thousand -- things seem to have levelled off by then
# Maximum number of steps per episode
max_steps_per_episode = 40

win_history = []

# Change this to False to skip visualisations during training - they slow everything down
show_vis = False

''' PERSISTING PARAMETERS '''
img_size = 4
num_channels = 3

explore = chance_explore

def conv_relu_layer(input, n_input, n_filters, filter_size, stride):
    weights = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, n_input, n_filters], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[n_filters]))
    conv_layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer += biases
    c_r_layer = tf.nn.relu(conv_layer)
    return c_r_layer

def maxpool_relu_layer(input):
    m_layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    m_r_layer = tf.nn.relu(m_layer)
    return m_r_layer

def flatten(input_layer):
    shape = input_layer.get_shape()
    num_features = shape[1:4].num_elements()
    flat_layer = tf.reshape(input_layer, [-1, num_features])
    return flat_layer

def fc_layer(input, n_inputs, n_outputs, use_relu=True):
    weights = tf.Variable(tf.truncated_normal(shape=[n_inputs, n_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[n_outputs]))
    fc_layer = tf.matmul(input, weights) + biases
    if use_relu:
        fc_layer = tf.nn.relu(fc_layer)
    return fc_layer

''' SET UP TENSORFLOW MODEL '''

with tf.device(device):
    g = tf.Graph()
    with g.as_default():

        state = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='state') # shape=[None, img_size, img_size, num_channels]
        q_s_a = tf.placeholder(tf.float32, shape=[None, env.num_actions])
        conv1 = conv_relu_layer(input=state, n_input=num_channels, n_filters=n_filters_conv1,
                                filter_size=filter_size_conv1, stride = stride1)
        max1 = maxpool_relu_layer(conv1)
        #conv2 = conv_relu_layer(input=max1, n_input=n_filters_conv1, n_filters=n_filters_conv2,
        #                        filter_size=filter_size_conv2, stride = stride2)
        #max2 = maxpool_relu_layer(conv2)
        #conv3 = conv_relu_layer(input=conv2, n_input=n_filters_conv2, n_filters=n_filters_conv3,
        #                        filter_size=filter_size_conv3, stride=stride3)
        #max3 = maxpool_relu_layer(conv3)
        flat = flatten(max1)
        fc1 = fc_layer(input=flat, n_inputs=flat.get_shape()[1:4].num_elements(), n_outputs=fc1_layer_size, use_relu=False)
        final = fc_layer(input=fc1, n_inputs=fc1_layer_size, n_outputs=env.num_actions, use_relu=False)
        loss = tf.losses.mean_squared_error(q_s_a, final)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        saver = tf.train.Saver()
        played_states = []
        played_rewards = []
        # session run with one kind of label
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in range(num_episodes):

                # Reset the environment and get the starting state
                #
                # The state is an env.H x env.W colour image with
                # green pixel corresponding to slippery ice,
                # blue pixel corresponding to water (R=-1)
                # red pixel corresponding to goal (R=1)
                # white pixel indicating the player's location
                s = env.reset()
                total_reward = 0
                #print(s)
                # Visualisation of the current state
                if show_vis:
                    env.show(s=s)
                #opt = sess.run(optimizer, feed_dict={states: s, q_s_a: y_batch})
                for i in range(max_steps_per_episode):
                    # If the environment got to the terminal state
                    # start a new episode and decrease likelihood of random exploration as it trains
                    if env.terminal():
                        explore -= 0.00001  # 1.0/((i/50)+(1/chance_explore))
                        explore = max(explore, 0.01)
                        break

                    # Pick a random action - this is where your policy
                    # would make some choices of actions - there are
                    # 4 actions index from 0 to 3 corresponding to
                    # player's movement towards N,E,S and W.
                    #
                    # ...
                    st = np.reshape(s,[1,4,4,3])
                    q_list = sess.run(final, feed_dict={state: st})
                    a = np.argmax(q_list)
                    if np.random.rand(1) < explore:
                        a = np.random.randint(0, env.num_actions)
                    old_s = st
                    # Execute the action, get the next state and the reward
                    s, R = env.step(a)

                    # Here you need to do something where you use the reward
                    # to update your policy in online mode, or or store the state reward and
                    # action for later update to the policy if you're training
                    # the policy in the batch mode
                    #
                    # ...
                    if online:
                        # Obtain the Q' values by feeding the new state through our network
                        st = np.reshape(s,[1,4,4,3])
                        q_list_next = sess.run(final, feed_dict={state: st})
                        # Obtain maxQ' and set our target value for chosen action.
                        max_q = np.max(q_list_next)
                        targetQ = q_list
                        #print(targetQ)
                        #print(a)
                        targetQ[0, a] = R + weight * max_q
                        # Train the network using target and predicted Q values
                        _ = sess.run(optimizer, feed_dict={state: old_s, q_s_a: targetQ})
                        total_reward += R

                    # Shows the new state of the environment - only the player's location
                    # might change
                    if show_vis:
                        env.show(s=s)

                # If you're training in batch mode, here's where you might want to
                # use the stored buffer of episode states,actions and rewards in order
                # to improve the policy
                #
                # ...
                played_states.append(i)
                played_rewards.append(total_reward)

                infoStr = "Episode %d, " % (e+1)
                if R==1:
                    infoStr += "win, "
                    win_history.append(1)
                else:
                    win_history.append(0)
                    if R==-1:
                        infoStr += "loss, "
                    else:
                        infoStr += "timeout, "

                if (e+1) % 100 == 0:
                    if (e+1) > 1000:
                        win_history_1000 = win_history[-1000:]
                        win_rate = np.sum(win_history_1000)/1000
                    else:
                        win_rate = np.sum(win_history) / len(win_history)
                    infoStr += "wins rate: %.2f \n" % win_rate
                    with open("results/"+id+".txt", 'a') as f:
                        f.write(infoStr)
                    #print(infoStr)
                    sys.stdout.flush()
                #if win_rate > 0.5:
                    #save_path = saver.save(sess, "trained/trained_" + str(win_rate))
            save_path = saver.save(sess, "trained/"+id+"_"+str(win_rate))

# Show the final score (ratio of wins over episodes)
env.show(blocking=True,id=id)
