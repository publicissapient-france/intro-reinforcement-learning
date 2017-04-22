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

    if FLAGS.load_previous_checkpoint:
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # Start training
    epsilon = FLAGS.epsilon_start

    t = 0
    while "flappy bird" != "angry bird":

        # Choose an action epsilon greedily
        readout_t = logits.eval(feed_dict={x: [s_t]})[0]
        a_t = np.zeros([FLAGS.nb_actions])
        action_index = 0
        if t % FLAGS.frame_per_action == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(FLAGS.nb_actions)
                a_t[random.randrange(FLAGS.nb_actions)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # Do nothing

        # Scale down epsilon
        if epsilon > FLAGS.epsilon_end and t > FLAGS.nb_time_steps_observe:
            epsilon -= (FLAGS.epsilon_start - FLAGS.epsilon_end) / FLAGS.nb_time_steps_explore

        # Run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # Store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > FLAGS.replay_memory:
            D.popleft()

        # Only train if done observing
        if t > FLAGS.nb_time_steps_observe:
            # sample a minibatch to train on
            minibatch = random.sample(D, FLAGS.batch_size)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = logits.eval(feed_dict={x: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # If terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + FLAGS.gamma * np.max(readout_j1_batch[i]))

            # Perform gradient step
            train_step.run(feed_dict={y: y_batch, a: a_batch, x: s_j_batch})

        # Update the old values
        s_t = s_t1
        t += 1

        # Save progress every 10000 iterations
        if t % FLAGS.checkpoint_frequency == 0:
            saver.save(sess, 'saved_networks/' + FLAGS.game + '-dqn', global_step=t)

        # Print info
        if t <= FLAGS.nb_time_steps_observe:
            state = "observe"
        elif FLAGS.nb_time_steps_observe < t <= FLAGS.nb_time_steps_observe + FLAGS.nb_time_steps_explore:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nb_time_steps_observe',
        type=float,
        default=10000.,
        help='Timesteps to observe before training'
    )
    parser.add_argument(
        '--nb_time_steps_explore',
        type=float,
        default=3000000.,
        help='Number of frames over which to anneal epsilon'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Decay rate of past observations'
    )
    parser.add_argument(
        '--epsilon_start',
        type=float,
        default=0.1,
        help='Starting value of epsilon'
    )
    parser.add_argument(
        '--epsilon_end',
        type=float,
        default=0.0001,
        help='Final value of epsilon'
    )
    parser.add_argument(
        '--replay_memory',
        type=int,
        default=50000,
        help='Number of previous transitions to remember'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Size of Mini-Batch'
    )
    parser.add_argument(
        '--frame_per_action',
        type=int,
        default=1
    )
    parser.add_argument(
        '--game',
        type=str,
        default="bird",
        help="Name of the game being played for log files"
    )
    parser.add_argument(
        '--nb_actions',
        type=int,
        default=2,
        help="Number of valid actions"
    )
    parser.add_argument(
        '--load_previous_checkpoint',
        type=bool,
        default=False,
        help="Whether to load previous checkpoints for training"
    )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=10000,
        help="Checkpoint save frequency"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
