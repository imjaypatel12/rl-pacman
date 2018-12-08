# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
# import random, time, util
from game import Directions
import game
from util import nearestPoint, Counter

from random import randrange, random as rand
import sys, math

agent_info = {}


#################
# Team creation #
#################

"""
Cheat code
"""

orig_stdout = sys.stdout
f = open('log.txt', 'w')
#sys.stdout = f


#sys.stdout = orig_stdout
#f.close()

def createTeam(indices, isRed, first='OffensiveReflexAgent', second='DefensiveReflexAgent', numTraining=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # XXX Assigns roles to the Pacman agents.
    agents = [eval('OffensiveReflexAgent' if (index//2)%2 == 0 else 'DefensiveReflexAgent')(index) for index in indices]
    # agents = [eval('OffensiveReflexAgent' if (index // 2) % 2 == 0 else 'DefensiveReflexAgent')(index) for index in
    #           indices]

    for a in agents:
        agent_info[a.index] = {'numReturned':0, 'numCarrying':0, 'totalFood': 0, 'totalFoodSet': False}

    return agents  # [eval('DefensiveReflexAgent')(index) for index in indices]

    # The following line is an example only; feel free to change it.
    # return [eval(first)(firstIndex), eval(second)(secondIndex)]

def normalize(listObj, min_, max_):
    w = np.array(listObj)
    w = np.interp(w, (w.min(), w.max()), (min_, max_))
    return w.tolist()

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    weights = {}
    rewardDiscount = 0.9
    training_n, nth, alpha, eps = (0, 0, 0.8, 0.5)

    def __init__(self, index):
        self.weights = self.getWeights(None, None)
        self.stepSize = 0.8
        self.rewardDiscount = 0.9
        super().__init__(index)

    @staticmethod
    def weighted_average(weight0Val: float, weight: float, weight1Val: float) -> float:
        return weight0Val * (1 - weight) + weight1Val * weight

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def updateWeights(self, gameState, action):
        """
        learn the weights
        """
        nextState = self.getSuccessor(gameState, action)
        nextAction = self._chooseAction_(nextState)
        reward = abs(self.getScore(nextState)) - abs(self.getScore(gameState))

        nextQValue = self.evaluate(nextState, nextAction)
        currentQValue = self.evaluate(gameState, action)

        agent_class = type(self)
        weights = agent_class.weights

        # self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)

        for (f, w) in weights.items():
            if f in features:
                weights[f] = w + DummyAgent.alpha \
                             * (reward + DummyAgent.rewardDiscount * nextQValue - currentQValue) \
                             * features[f]

        # to avoid the horror!
        # weights = dict(zip(weights.keys(), normalize(list(weights.values()), -1000, 1000)))
        # agent_class.weights = weights
        # print(features)
        # print(weights)
        # print("---")

    def _chooseAction_(self, gameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        return actions[randrange(len(actions))] if rand() < DummyAgent.eps \
            else self.computeAction(gameState, actions)

    def computeAction(self, gameState, actions):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())
        bestAction = None

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

        bestAction = bestAction if bestAction else random.choice(bestActions)

        # update weights
        self.updateWeights(gameState, bestAction)

        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return self.weights

    def getRole(self, agent):
        """
        Returns the subclass name to identify the role of an agent
        """
        return agent

    def getReward(self, gameState, action):
        """
        Return the reward for taking an action given the game state
        """
        return 1


class OffensiveReflexAgent(DummyAgent):
    # weights = {'successorScore': 100, 'foodRemaining': -1, 'distanceToFood': -.1, 'ghostDistance': .5,
    #             'distanceToSpawn': .5, 'pelletsCarrying': .9}
    weights = {'successorScore': 99.8856, 'foodRemaining': -0.4694, 'distanceToFood': -0.6182, 'ghostDistance': 5,
     'distanceToSpawn': -1.1, 'pelletsCarrying':0.6}#, 'loadRatio': -1}

    episode = DummyAgent.nth

    def getWeights(self, gameState, action):
        return OffensiveReflexAgent.weights

    def starting_state(self, gameState, action):
        i = agent_info[self.index]
        food_difference = len(self.getFood(self.getSuccessor(gameState, action)).asList()) + i['numReturned'] + i['numCarrying']
        return i['totalFood'] < food_difference

    def getFeatures(self, gameState, action):
        global agent_info

        if OffensiveReflexAgent.episode != DummyAgent.nth: #self.starting_state(gameState,action):
            OffensiveReflexAgent.episode = DummyAgent.nth
            print(agent_info[self.index])
            print(len(self.getFood(self.getSuccessor(gameState, action)).asList()))
            agent_info[self.index] = {'numReturned': 0,
                                      'numCarrying': 0,
                                      'totalFood': len(self.getFood(self.getSuccessor(gameState, action)).asList()),
                                      'totalFoodSet': True}
            print('Resetting carry num and return num...\n\n')

        feature_names = ['successorScore', 'foodRemaining', 'distanceToFood', 'ghostDistance',
                         'distanceToSpawn', 'pelletsCarrying', 'totalFood', 'run']
        features = util.Counter()
        for name in feature_names:
            features[name] = 0

        successor = self.getSuccessor(gameState, action)
        is_pacman = successor.getAgentState(self.index).isPacman
        myPos = successor.getAgentState(self.index).getPosition()  # (X,Y)
        features['successorScore'] = self.getScore(successor)

        foodList = self.getFood(successor).asList()
        features['foodRemaining'] = len(foodList)

        # default value of distance to nearest food to -1
        features['distanceToFood'] = -1
        if foodList:
            minDistToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistToFood

        opponentAgents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in opponentAgents if not a.isPacman and a.getPosition()]

        # if there are ghosts
        features['ghostDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) \
            if ghosts and is_pacman else -1

        features['distanceToSpawn'] = self.getMazeDistance(myPos, gameState.getInitialAgentPosition(self.index))
        features['run'] = 1 if (features['ghostDistance'] < features['distanceToFood']) and features['ghostDistance'] \
                               > 0 else 0
        info = agent_info[self.index]
        if is_pacman:
            info['numCarrying'] = info['totalFood'] - (features['foodRemaining'] + info['numReturned'])
            features['pelletsCarrying'] = info['numCarrying']

        elif info['numCarrying'] > 0:
            info['numReturned'] = info['numCarrying'] if features['distanceToSpawn'] != 0 else info['numReturned']
            info['numCarrying'] = 0
            print('Returned', info['numReturned']) if features['distanceToSpawn'] != 0 else print('Agent was Killed')

        # print('remaining', len(foodList), end='')
        # print('vars', gameState.getAgentState(self.index).__dict__)

        features = Counter(dict(zip(features.keys(), normalize(list(features.values()), 0, 1))))

        if info['numCarrying'] > 0: print(features)
        return features


class DefensiveReflexAgent(DummyAgent):
    weights = {'numInvaders': -3, 'onDefense': 1, 'invaderDistance': -1, 'stop': -1, 'reverse': -1}

    def getWeights(self, gameState, action):
        return DefensiveReflexAgent.weights

    def getFeatures(self, gameState, action):
        feature_names = ['numInvaders', 'onDefense', 'invaderDistance', 'stop', 'reverse']
        features = util.Counter()
        for name in feature_names:
            features[name] = 0

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1 if myState.isPacman else 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        features['numInvaders'] = len(invaders)
        # Distance to nearest invader. Large if no invaders.
        features['invaderDistance'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders]) \
            if invaders else -1

        features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0

        features = Counter(dict(zip(features.keys(), normalize(list(features.values()), 0, 1))))

        return features

