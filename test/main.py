# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:04:12 2020

@author: 2920029
"""

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import env.smallVaringGoalRep1Environment
env = gym.make('SmallVaringGoalRep1Env-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
enviroment = gym.make("SmallVaringGoalRep1Env-v0").env
enviroment.render()

print('Number of states: {}'.format(env.observation_space))
print('Number of actions: {}'.format(env.action_space.n))
env.render()

alpha = 0.1
gamma = 0.6
epsilon = 0.1

Q = LinearApproximatorOfActionValuesWithTile(alpha=0.0125, stateLow=env.stateLow, stateHigh=env.stateHigh, numActions=env.numActions)
epsilonGreedyPolicy = EpsilonGreedy(Q, 0.05)
greedyPolicy = EpsilonGreedy(Q, 0.)

numEpisodes = 50000

print("Initiating Learning with Q-learning with function approximation")

for episode in range(numEpisodes):
    # Initialize S, A, and done=False
    state = env.reset()
    reward = 0.
    done = False

    while not done:
        # Choose A from Q epsilon-greedily
        action = epsilonGreedyPolicy.action(state)
        # Take A and observe R and S'.
        nextState, reward, done = env.step(action)
        env.render()
        # Compute Q-learning target (R + gamma * max_a Q(s', a))
        target = reward + env.gamma * np.max(Q(nextState))
        # Update our function approximator
        Q.update(state, action, target)

        state = nextState

    print(str(episode) + "th episode: reward = " + str(reward))

    if episode % 1000 == 0:
        evaluation(env, greedyPolicy, str(episode))