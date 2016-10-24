from ale_python_interface import ALEInterface
import numpy as np
import cv2
import Agent_DQN as AI
import time
game = 'pong'

rom_file = 'roms/' + game + '.bin'
ale = ALEInterface()
ale.setInt('random_seed', np.random.randint(10000000))
width, height = [84, 84]
buffer_size = 4
skip = 4
ale.setInt('frame_skip', skip)
ale.setBool('display_screen', False)
ale.loadROM(rom_file)
actions = ale.getMinimalActionSet()

def random_action():
    ale.act(actions[np.random.randint(len(actions))])
    screen = ale.getScreenGrayscale()
    return screen

buf = np.zeros((buffer_size, width, height))


def act(action):
    global buf
    reward = ale.act(actions[np.random.randint(len(actions))])
    screen = ale.getScreenGrayscale()
    buf = np.roll(buf, 1, axis=0)
    buf[0] = cv2.resize(screen, (width, height)).reshape((1, width, height))
    return reward, ale.game_over()


agent = AI.GameAgent((buffer_size, width, height), len(actions), gamma=0.99, buffer_max_size=1000000,
                     save_name=game + '_simple',
                     PRIORITIZED_XP_REPLAY=False, DOUBLE_NETWORK=False, backup_steps=10000, debug_steps=500,
                     learning_rate=0.00025, DUELING_ARCHITECTURE=False, batch_size=32, learning_time=1000000,
                     train_every_steps=4)

EPISODES_TO_TEST = 5
GAMES_LIMIT = 500000
MAX_LEN = 1000000

sum = 0
for episode in xrange(GAMES_LIMIT):
    # Test every 30 episodes
    index = 0

    if episode % 100 == 0 and episode > -1:

        for i in xrange(EPISODES_TO_TEST):
            ale.reset_game()
            buf = np.zeros((buffer_size, width, height))
            total_reward = 0.
            for _ in xrange(MAX_LEN):
                if index > 30:
                    action = agent.e_greedy_action([np.copy(buf) / 255.], 0.01)
                else:
                    action = np.random.randint(len(actions))
                reward, done = act(action)
                total_reward += reward
                index += 1
                if done:
                    break
            print 'episode: ', episode, 'Test reward:', total_reward
        print 'Mean total scores of the train episodes is ', float(sum) / 100
        sum = 0

    ale.reset_game()
    buf = np.zeros((buffer_size, width, height))

    # Train
    total_reward = 0.
    index = 0
    for _ in xrange(MAX_LEN):

        if index > 30:
            action = agent.action([np.copy(buf) / 255.], episode)
        else:
            action = np.random.randint(len(actions))

        reward, done = act(action)
        total_reward += reward
        reward = np.clip(reward, -1, 1)
        this_b = np.copy(buf)
        if index >= 30:  # Because this_buf contains last 4 frames
            agent.memorize(this_b, action, float(reward), done)
        if agent.time_step % 1000 == 1:
            print agent.time_step
        index += 1
        if done:
            break
    sum += total_reward

    if episode % 1 == 0:
        print 'Reward of episode ' + str(episode) + ' is ' + str(total_reward)
