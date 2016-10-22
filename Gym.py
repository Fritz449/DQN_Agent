import gym
import numpy as np
import time
import Agent_DQN as AI
import cv2

ENV_NAME = 'Breakout-v0'
np.set_printoptions(threshold=np.inf)
env = gym.make(ENV_NAME)

ATARI = True

state_dim = env.observation_space.shape
print state_dim
action_dim = env.action_space.n
print action_dim
if ATARI:
    state_shape = (4, 80, 80)
else:
    state_shape = state_dim
agent = AI.GameAgent(state_shape, action_dim, gamma=0.99, buffer_max_size=1000000, save_name=ENV_NAME + '_simple',
                     PRIORITIZED_XP_REPLAY=False, DOUBLE_NETWORK=False, backup_steps=10000, debug_steps=100,
                     learning_rate=0.00025, DUELING_ARCHITECTURE=False, batch_size=32, learning_time=1000000,
                     train_every_steps=4)
EPISODES_TO_TEST = 1
GAMES_LIMIT = 500000
MAX_LEN = 1000000


# Preprocessing for atari
def atari_prep(img):
    img = cv2.resize(cv2.cvtColor(img,
                                  cv2.COLOR_RGB2GRAY), (80, 80))
    img = np.transpose(img.astype(np.uint8))

    return img


# Make next 4-images concatenation by shifting old images
def next_buf(buffer, gray):
    buf = np.copy(buffer)
    buf[0, :, :] = np.copy(buf[1, :, :])
    buf[1, :, :] = np.copy(buf[2, :, :])
    buf[2, :, :] = np.copy(buf[3, :, :])
    buf[3, :, :] = np.copy(gray)
    del buffer
    return buf


sum = 0
for episode in xrange(GAMES_LIMIT):
    # Test every 30 episodes
    index = 0

    if episode % 100 == 0 and episode > 0:
        total_reward = 0

        for i in xrange(EPISODES_TO_TEST):
            state = env.reset()

            if ATARI:
                this_buf = np.zeros(state_shape, dtype=np.uint8)
                this_buf = next_buf(this_buf, atari_prep(state))

            for _ in xrange(MAX_LEN):
                env.render()
                if ATARI:
                    if index > 30:
                        action = agent.e_greedy_action([np.copy(this_buf) / 255.], 0.04)
                    else:
                        action = np.random.randint(action_dim)
                else:
                    action = agent.e_greedy_action([state], 0.04)
                next_state, reward, done, _ = env.step(action)
                state = np.copy(next_state)

                if ATARI:
                    this_buf = next_buf(this_buf, atari_prep(next_state))

                total_reward += reward
                index += 1
                if done:
                    break
        ave_reward = total_reward / EPISODES_TO_TEST
        print 'episode: ', episode, 'Average Reward:', ave_reward
        print 'Mean total scores of the train episodes is ', float(sum)/100
        sum = 0

    state = env.reset()

    if ATARI:
        this_buf = np.zeros(state_shape, dtype=np.uint8)
        this_buf = next_buf(this_buf, atari_prep(state))

    # Train
    total_reward = 0
    index = 0
    # if episode % 10 == 0:
    #     print "Now agent plays episode number " + str(episode)
    for _ in xrange(MAX_LEN):

        if ATARI:
            if index > 30:
                action = agent.action([np.copy(this_buf) / 255.], episode)
            else:
                action = np.random.randint(action_dim)
        else:
            action = agent.action([state], episode)

        next_state, reward, done, info = env.step(action)
        if ATARI:
            if index >= 30:  # Because this_buf contains last 4 frames
                agent.memorize(np.copy(this_buf), action, float(reward), done)

            this_buf = next_buf(this_buf, atari_prep(next_state))

        else:
            agent.memorize(np.copy(state), action, float(reward), done)
        if agent.time_step % 1000 == 1:
            print agent.time_step
        total_reward += reward

        state = np.copy(next_state)
        index += 1
        if done:
            break
    sum += total_reward

    if episode % 1 == 0:
        print 'Reward of episode ' + str(episode) + ' is ' + str(total_reward)
