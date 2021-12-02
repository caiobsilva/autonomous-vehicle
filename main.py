# Caio Braga, Juan Suman, Matheus Eiji

import gym, sys, os, time, random
import numpy as np

def animate(frames):
  clear_console = 'clear' if os.name == 'posix' else 'CLS'

  while True:
    for frame in frames:
      os.system(clear_console)
      sys.stdout.write(frame)
      sys.stdout.flush()
      time.sleep(0.1)

env = gym.make('gym_parking:parking_lot-v0')
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

frames = []
itr = 100000

for i in range(1, itr + 1):
  state = env.reset()
  reward = 0
  done = False

  while not done:
    if random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()
    else:
      action = np.argmax(q_table[state])

    next_state, reward, done, info = env.step(action)

    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state, action] = new_value

    state = next_state

    if i == itr:
      frames.append(env.render(mode='ansi'))

print("Training finished.\n")
animate(frames)
