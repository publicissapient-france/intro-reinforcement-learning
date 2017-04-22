#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import cv2
import random
import numpy as np
from collections import deque

import game.wrapped_flappy_bird as game
from network import DeepConvNetwork


def main(_):
    sess = tf.InteractiveSession()

    # Create Network
    deep_conv_network = DeepConvNetwork(nb_actions=FLAGS.nb_actions)
    x, logits = deep_conv_network.create_network()

    # Define the cost function
    a = tf.placeholder("float", [None, FLAGS.nb_actions])
    y = tf.placeholder("float", [None])
    logit_action = tf.reduce_sum(tf.multiply(logits, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - logit_action))

    # Train step
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Open up a game state to communicate with emulator
    game_state = game.GameState()

    # Store the previous observations in replay memory
    D = deque()

    # Get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(FLAGS.nb_actions)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_directory)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    t = 0
    while "flappy bird" != "angry bird":

        # Choose an action
        readout_t = logits.eval(feed_dict={x: [s_t]})[0]
        a_t = np.zeros([FLAGS.nb_actions])

        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        # Run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # Update the old values
        s_t = s_t1
        t += 1

        state = "play"

        print("TIMESTEP", t, "/ STATE", state,
              "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_directory',
        type=str,
        default="saved_networks",
        help="Directory containing the checkpoints to load"
    )
    parser.add_argument(
        '--nb_actions',
        type=int,
        default=2,
        help="Number of valid actions"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
