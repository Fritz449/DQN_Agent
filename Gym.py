from scipy.misc import imresize

import gym
import numpy as np

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
agent = AI.GameAgent(state_shape, action_dim, gamma=0.99, buffer_max_size=50000, save_name=ENV_NAME,
                     FREEZE_WEIGHTS=True, PRIORITIZED_XP_REPLAY=True, DOUBLE_NETWORK=True, freeze_steps=3000)


EPISODES_TO_TEST = 1
GAMES_LIMIT = 100000
import time

hsh = np.random.randint(0, 10, size=(4, 84, 84))


def ghash(state):
    return np.sum(state / 100. * hsh)


def atari_prep(img):
    # print img
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    del r, g, b
    gray = imresize(gray, (84, 84))
    return (gray / 256.).reshape(1, 84, 84).astype('float32')


def next_buf(buffer, gray):
    buf = np.copy(buffer)
    buf[0, :, :] = buf[1, :, :]
    buf[1, :, :] = buf[2, :, :]
    buf[2, :, :] = buf[3, :, :]
    buf[3, :, :] = gray
    return buf


for episode in xrange(GAMES_LIMIT):
    # initialize task
    if episode % 1 == 0 and episode > 0:
        total_reward = 0
        for i in xrange(EPISODES_TO_TEST):
            state = env.reset()
            if ATARI:
                this_buf = np.zeros((4, 84, 84))
                this_buf = next_buf(this_buf, atari_prep(state))

            while True:
                env.render()
                if ATARI:
                    action = agent.greedy_action([this_buf])
                else:
                    action = agent.greedy_action([state])
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if ATARI:
                    this_buf = next_buf(this_buf, atari_prep(next_state))
                total_reward += reward
                time.sleep(0.0001)
                if done:
                    break
        ave_reward = total_reward / EPISODES_TO_TEST
        print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward

    state = env.reset()
    if ATARI:
        this_buf = np.zeros((4, 84, 84))
        this_buf = next_buf(this_buf, atari_prep(state))
    # Train
    total_reward = 0
    index = 0
    if episode % 50 == 0:
        print episode
    while True:
        if ATARI:
            if episode % 2 == 0:
                action = agent.greedy_action([this_buf])  # greedy action for train
            else:
                action = agent.e_greedy_action([this_buf])  # e-greedy action for train
        else:
            if episode % 2 == 0:
                action = agent.greedy_action([state])  # greedy action for train
            else:
                action = agent.e_greedy_action([state])  # e-greedy action for train

        next_state, reward, done, info = env.step(action)
        if agent.buffer_size % 1000 == 1:
            print agent.buffer_size

        # if index >= 4:
        if ATARI:
            if index >= 4:
                agent.memorize(this_buf, action, float(reward), done)
            this_buf = next_buf(this_buf, atari_prep(next_state))
        else:
            agent.memorize(state, action, float(reward), done)

        total_reward += reward
        state = next_state
        index += 1
        if done:
            break
    print total_reward
    # Test every 100 episodes
