'''
V5 - Per user offers

author Matthew Schofield
version 12.2.2020
'''

import gym
from gym import spaces
import numpy as np
from copy import deepcopy
import math
import random

class BSS_Controller_User(gym.Env):
    '''
    '''

    def __init__(self, systemInitObj, budget, stepFile):
        """
        Initializes the environment
        """
        # Total number of regions
        self.systemSize = systemInitObj.getSystemSize()
        self.steps = systemInitObj.getStepsPerEpisode()
        self.stepFile = stepFile

        # Static Inits
        # Budget - the budget the agent has access to
        # S - Supply of bikes at each region, ie S[i] is current the supply at region i
        # A - Arrivals this timestep
        # D - Departures this timestep
        # U - Unservice level this timestep
        # E - Expense for each region last timestep
        # Tdij - Tensor for intended movements region i to j, matrix (i,j) number from i=>j
        self.staticInits = {
            "Steps": self.steps,
            "Budget": budget,
            "S": systemInitObj.getInitSupply(),
            "A": np.zeros((self.systemSize**2,)),
            "D": np.zeros((self.systemSize**2,)),
            "U" : np.zeros((self.systemSize**2,)),
            "E" : np.zeros((self.systemSize**2,)),
            "Tdij": systemInitObj.getUsers()
        }

        self.accumulatedRew = 0

        # OpenAi Gym settings
        self.action_space = spaces.Box(0, 5, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 100.0, shape=(11, self.systemSize,), dtype=np.float32)

        # Revert to original values
        self.revertToStaticInits()

    def getBudget(self):
        return self.budget

    '''
    Gym top-levels
    
    Up
    Down
    Left
    Right
    '''
    def step(self, action):
        # Move user
        neighbors = [n for n in self.neighbors(self.userA)]
        utilities = []
        for i, n in enumerate(neighbors):
            if n == -1:
                utilities.append((i, -1, n))
            else:
                utilities.append((i, action[i] - self.walkingCost(), n))
        # Find 'best' move
        maxNeighbor = -1
        maxUtil = -1
        direction = 0
        for i, u, n in utilities:
            if u > maxUtil and self.S[n] > 0:
                maxUtil = u
                maxNeighbor = n
                direction = i
        # Check that the move is acceptable
        if maxUtil >= 0 and self.budget >= action[direction]:
            # Service
            self.D[maxNeighbor] += 1
            self.A[self.userD] += 1
            self.S[maxNeighbor] -= 1
            self.budget -= action[direction]
            reward = 1
        else:
            # Miss service
            reward = 0
        keepGoing = True
        while self.curStep < 24 and keepGoing:
            while self.userIter < self.hourLen:
                user = self.Tdij[self.curStep][self.userIter]
                self.serviceLevel += 1
                self.userA = user[0]
                self.userD = user[1]
                self.userIter += 1
                if self.S[self.userA] > 0:
                    self.S[self.userA] -= 1
                    self.D[self.userA] += 1
                    self.A[self.userD] += 1
                else:
                    keepGoing = False
                    break
            if keepGoing:
                self.curStep += 1
                self.userIter = 0
                if self.curStep == 24:
                    break
                else:
                    self.hourLen = len(self.Tdij[self.curStep])
                    self.resolvePreviousHour()

        # Format output
        state = self.buildState()

        done = self.curStep == self.steps
        info = {}
        self.accumulatedRew += reward
        self.requested += 1
        return state, reward, done, info

    def reset(self):
        '''
        Reset environment

        :return: New state
        '''
        self.requested = max(self.requested, 1)
        self.serviceLevel = max(self.serviceLevel, 1)
        decreased = 1 - ((self.requested - self.accumulatedRew) / self.requested)
        serviceLevel = ((self.serviceLevel - self.requested) + self.accumulatedRew)/self.serviceLevel
        self.stepFile.write(str("%.3f,%.3f,%s\n" % (decreased, serviceLevel, str(self.budget))))
        self.accumulatedRew = 0
        self.serviceLevel = 0

        self.revertToStaticInits()
        return self.buildState()

    def render(self, mode='human'):
        '''
        Render environment

        :param mode: changes representation method
            human -> print to terminal state representation
        '''
        print("Current Counts:")
        print(self.S)

    def close(self):
        '''
        Clean up environment

        Rarely called
        '''
        self.stepFile.close()
        print("Closed")

    '''
    Helper methods
    '''
    def buildState(self):
        state = np.vstack([
            self.S.reshape((self.systemSize, self.systemSize)),
            np.array([self.userA, self.userA, self.userA, self.userA, self.userA, self.userA,
                      self.userD, self.userD, self.userD, self.budget/self.staticInits["Budget"]])
        ])
        return state

    def walkingCost(self):
        # alpha
        alpha = 1.0
        userLoc = random.random()
        bikeLoc = 1 + random.random()

        return alpha * (bikeLoc - userLoc)**2

    def resolvePreviousHour(self):
        # Resolve arrivals
        self.S = np.add(self.S, self.A)

        # Shift time window
        self.A_1 = deepcopy(self.A)
        self.D_1 = deepcopy(self.D)
        self.E_1 = deepcopy(self.E)
        self.U_1 = deepcopy(self.U)

        # Reset
        self.A = deepcopy(self.staticInits["A"])
        self.D = deepcopy(self.staticInits["D"])
        self.E = deepcopy(self.staticInits["E"])
        self.U = deepcopy(self.staticInits["U"])


    def revertToStaticInits(self):
        '''
        Properly resetting the environment is one of the trickiest components in writing RL simulation software
        This routine helps to ensure proper memory management
        '''
        # This is only an int so it does not necessarily need a deepcopy, but this is not noticeably expensive
        # And adds resistance to future changes where Budget may be a 'real' Object
        self.budget = deepcopy(self.staticInits["Budget"])
        self.S = deepcopy(self.staticInits["S"])
        self.A = deepcopy(self.staticInits["A"])
        self.D = deepcopy(self.staticInits["D"])
        self.U = deepcopy(self.staticInits["U"])
        self.E = deepcopy(self.staticInits["E"])
        # A(t-1), D(t-1)
        self.A_1 = deepcopy(self.staticInits["A"])
        self.D_1 = deepcopy(self.staticInits["D"])
        self.U_1 = deepcopy(self.staticInits["U"])
        self.Tdij = deepcopy(self.staticInits["Tdij"])
        self.E_1 = deepcopy(self.staticInits["E"])
        self.curStep = 0

        self.userA = 0
        self.userD = 0
        self.userIter = 0
        self.requested = 0
        self.serviceLevel = 0
        self.hourLen = len(self.Tdij[self.curStep])

    def neighbors(self, regionI):
        '''
        Outputs neighbors of index regionI

        :param regionI: index of region whose neighbors are to be found
        :return: [up, down, left, right]
        '''
        # Get length of rows, assumes square layout
        rowLen = self.systemSize

        up = regionI - rowLen
        down = regionI + rowLen
        left = regionI - 1
        right = regionI + 1

        if up < 0:
            up = -1
        if down >= self.systemSize**2:
            down = -1
        if left % rowLen == rowLen-1:
            left = -1
        if right % rowLen == 0:
            right = -1

        return [int(up), int(down), int(left), int(right)]


