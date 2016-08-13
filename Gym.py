from scipy.misc import imresize

import gym
import numpy as np
import time
import Agent_DQN as AI

ENV_NAME = 'CartPole-v0'
np.set_printoptions(threshold=np.inf)
env = gym.make(ENV_NAME)

ATARI = False

state_dim = env.observation_space.shape
print state_dim
action_dim = env.action_space.n
print action_dim
if ATARI:
    state_shape = (4, 84, 84)
else:
    state_shape = state_dim
agent = AI.GameAgent(state_shape, action_dim, gamma=0.99, buffer_max_size=28000, save_name=ENV_NAME,
                     PRIORITIZED_XP_REPLAY=True, DOUBLE_NETWORK=True, backup_steps=3000, debug_steps=100,
                     learning_rate=0.0001)

EPISODES_TO_TEST = 1
GAMES_LIMIT = 100000
MAX_LEN = 10000000


# Preprocessing for atari
def atari_prep(img):
    # print img
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    del r, g, b
    gray = imresize(gray, (84, 84))
    return (gray / 256.).reshape(1, 84, 84).astype('float32')


# Make next 4-images concatenation by shifting old images
def next_buf(buffer, gray):
    buf = np.copy(buffer)
    buf[0, :, :] = np.copy(buf[1, :, :])
    buf[1, :, :] = np.copy(buf[2, :, :])
    buf[2, :, :] = np.copy(buf[3, :, :])
    buf[3, :, :] = np.copy(gray)
    del buffer
    return buf


for episode in xrange(GAMES_LIMIT):
    # Test every 30 episodes
    if episode % 30 == 0 and episode > 0:
        total_reward = 0
        for i in xrange(EPISODES_TO_TEST):
            state = env.reset()

            if ATARI:
                this_buf = np.zeros((4, 84, 84))
                this_buf = next_buf(this_buf, atari_prep(state))

            for _ in xrange(MAX_LEN):
                env.render()
                if ATARI:
                    action = agent.greedy_action([this_buf])
                else:
                    action = agent.greedy_action([state])
                next_state, reward, done, _ = env.step(action)
                state = np.copy(next_state)

                if ATARI:
                    this_buf = next_buf(this_buf, atari_prep(next_state))

                total_reward += reward
                time.sleep(0.0001)
                if done:
                    break
        ave_reward = total_reward / EPISODES_TO_TEST
        print 'episode: ', episode, 'Average Reward:', ave_reward

    state = env.reset()

    if ATARI:
        this_buf = np.zeros((4, 84, 84))
        this_buf = next_buf(this_buf, atari_prep(state))

    # Train
    total_reward = 0
    index = 0
    # if episode % 10 == 0:
    #     print "Now agent plays episode number " + str(episode)
    for _ in xrange(MAX_LEN):
        if ATARI:
            action = agent.action([this_buf], episode)  # greedy or e-greedy depends on number of episode
        else:
            action = agent.action([state], episode)  # greedy or e-greedy depends on number of episode

        next_state, reward, done, info = env.step(action)
        if ATARI:
            if index >= 3:  # Because this_buf contains last 4 frames
                agent.memorize(np.copy(this_buf), action, reward, done)
            this_buf = next_buf(this_buf, atari_prep(next_state))
        else:
            agent.memorize(np.copy(state), action, float(reward), done)
        # if agent.time_step % 1000 == 1:
        #     print agent.time_step
        total_reward += reward
        state = next_state
        index += 1
        if done:
            break

    # print 'Reward of episode ' + str(episode) + ' is ' + str(total_reward)
