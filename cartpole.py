import random as rand
import numpy as np
import gym
import sys

class SemiGradientSarsaAgent:

    def __init__(self, stateSize, actions, explorationRate, stepSize, rewardDiscount):
        self.actions = np.array(actions)
        self.weights = np.zeros((len(actions), stateSize), dtype=np.float32)
        self.explorationRate = explorationRate
        self.stepSize = stepSize
        self.rewardDiscount = rewardDiscount

    def chooseAction(self, state):
        return self.__chooseActionInternal__(state)[1]

    def learn(self, state, action, newState, reward, done):
        prevQValue = self.__getQValue__(state, action)
        if done:
            step = self.stepSize * (reward - prevQValue)
        else:
            nextQValue = self.__chooseActionInternal__(newState)[0]
            step = self.stepSize * (reward + self.rewardDiscount * nextQValue - prevQValue)

        # Gradient descent: The Gradient for the weights is the state itself
        gradient = state
        self.weights[action] += step * gradient

    def __chooseActionInternal__(self, state):
        # Epsilon Greedy
        qValues = np.array([self.__getQValue__(state, action) for action in self.actions])
        if rand.random() < self.explorationRate:
            index = rand.choice(range(len(self.actions)))
        else:
            index = np.argmax(qValues)

        return (qValues[index], self.actions[index])

    # Calculate the QValue with a linear function
    def __getQValue__(self, state, action):
        return np.sum(state * self.weights[action])

def trainAgent(agent, episodes = 1):
    env = gym.make('CartPole-v1')
    totalRewards = []
    for i in range(episodes):
        sys.stdout.write("\rEpisode {}\\{}".format(i + 1, episodes))
        state = env.reset()
        # throw out the position
        state = np.array([*state[1:]]) 
        totalReward = 0
        # Run one episode and train the agent
        while True:
            action = agent.chooseAction(state)
            newState, reward, done, _ = env.step(action)
            # throw out the position
            newState = np.array([*newState[1:]]) 

            totalReward += reward
            agent.learn(state, action, newState, reward, done)

            if done:
                break

            state = newState

        totalRewards.append(totalReward)

    return totalRewards

agent = SemiGradientSarsaAgent(3, np.array([0, 1]), 0.1, 0.8, 1)
rewards = trainAgent(agent, 3000)