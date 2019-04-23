import gym
import gym_roos
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
import pdb

env = gym.make('SaccadeMultDigits-v0')
env.render()

while True:
	a = np.random.rand(21) * 2 - 1
	
	ta = input('Action: (1) Saccade, (2) Classify, (3) Uncertain, (4) Done: ')
	if ta=='':
		a[0] = 1.0  # saccade
	else:
		a[int(ta)-1] = 1.0

	loc = input('Location (x, y): ')
	if loc!='':
		xy = loc.split(',')
		x = float(xy[0])
		y = float(xy[1])
		a[4] = x
		a[5] = y

	label = input('Class: ')
	if label != '':
		label = int(label)
		a[10+label] = 1

	state, reward, done, _ = env.step(a)
	env.render()

	pdb.set_trace()
