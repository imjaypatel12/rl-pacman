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


#################
# Team creation #
#################

# def createTeam(firstIndex, secondIndex, isRed,
#               first = 'DummyAgent', second = 'DummyAgent'):
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
    # agents = [eval('OffensiveReflexAgent' if (index//2)%2 == 0 else 'DefensiveReflexAgent')(index) for index in indices]
    agents = [eval('OffensiveReflexAgent' if (index // 2) % 2 == 0 else 'DefensiveReflexAgent')(index) for index in
              indices]

    return agents

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

    @staticmethod
    def weighted_average(weight0Val:float, weight:float, weight1Val:float) -> float:
        return weight0Val*(1 - weight) + weight1Val*weight

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
        
        #self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)

        for (f, w) in weights.items():
            if f in features:
                weights[f] = w + DummyAgent.alpha \
                            * (reward + DummyAgent.rewardDiscount * nextQValue - currentQValue) \
                            * features[f]

        # to avoid the horror!
        #weights = dict(zip(weights.keys(), normalize(list(weights.values()), -1000, 1000)))
        #agent_class.weights = weights
        #print(features)
        #print(weights)
        #print("---")


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


class OffensiveReflexAgent(DummyAgent):
    weights = {'foodRemaining': 1, 'successorScore': 1, 'distanceToFood': 1, 'run': 1, 'ghostDistance': 1,
                'stop': 1, 'reverse': 1}

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        #features['successorScore'] = self.getScore(successor)
        # print("Score", features['successorScore'])

        foodList = self.getFood(successor).asList()
        features['foodRemaining'] = -len(foodList)

        # default value of distance to nearest food to +infinity
        features['distanceToFood'] = minDistToFood = -9999
        if foodList:
            minDistToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = -minDistToFood

        opponentAgents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in opponentAgents if not a.isPacman and a.getPosition()]

        features['ghostDistance'] = -9999
        if ghosts:  # if there are ghosts
            distsToGhosts = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
            features['ghostDistance'] = -min(distsToGhosts)

            # run(1) if ghosts are closer than food, else don't run(0).
            #features['run'] = 1 if features['ghostDistance'] < minDistToFood else 0

        #features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        #features['reverse'] = 1 if action == rev else 0

        features = Counter(dict(zip(features.keys(), normalize(list(features.values()), 0, 1))))

        return features

    def getWeights(self, gameState, action):
        return OffensiveReflexAgent.weights


class DefensiveReflexAgent(DummyAgent):
    weights = {'numInvaders': 1, 'onDefense': 1, 'invaderDistance': 1, 'stop': 1, 'reverse': 1}

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
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = - len(invaders)
        # Distance to nearest invader. Infinite if no invaders.
        features['invaderDistance'] = - min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders]) \
            if invaders else 0

        # TODO Figure out what these do lol
        #features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        #features['reverse'] = 1 if action == rev else 0

        features = Counter(dict(zip(features.keys(), normalize(list(features.values()), 0, 1))))
        return features

    def getWeights(self, gameState, action):
        return DefensiveReflexAgent.weights
