'''
V1

Spec detailed in class string

author Matthew Schofield
version 11.27.2020
'''
import gym
from gym import spaces
import numpy as np
from copy import deepcopy
import random

class BSS_Controller_Base(gym.Env):
    '''
    Author Matthew Schofield
    Version 11.16.2020

    Implementation of a BikeShare Environment

    An environment inspired by the framework from the paper:
    A Deep Reinforcement Learning Framework for Rebalancing Dockless Bike Sharing Systems

    Original data could not be located so this environment will allow for distributions to be 'plugged-in'

    It should also be noted that the original paper relied on one month of Shanghai Mobike data
    Aug 1st - Sept 1st 2016.

    Paper Environment Implementation Summary:

    Definitions:
    Area n Regions: R={r1, r2, ..., rn}
    Day discretized into timeslots: T={t1, t2, ..., tm}
    N(ri) = ri's neighbors

    Si(t) = Supply in region ri at beginning of timeslot t
    S(t) = Vector of all region supplies at time t = (Si(t), {for all}i)
    Ai(t) = Arrivals in region ri at timeslot t
    A(t) = Vector of all region arrivals at time t = (Ai(t), {for all}i)
    Di(t) = Demand in region ri at timeslot t
    D(t) = Vector of all region demands at time t = (Di(t), {for all}i)
    dij(t) = Number of users intending to ride from region ri to region rj during timeslot t

    Pricing Algorithm:
    A - Agent - If a user arrives at ri in time tx and there are no bikes in ri the agent A can recommend bikes in N(ri)
        A will also give price incentive pij(t), offer at time t to move i to j
    B - Budget that the agent can use

    User Model:
    For all users in region ri, if there are available bikes they will take those. Otherwise apply walking cost model,
    ck(i,j,x) = { 0     i == j
                { ax^2  j in N(ri)
                { +inf  else
    uk(i, j, t, x) = utility of offer to user = pij(t) - ck(i,j,x)
    The user will take the max non-negative uk(i, j, t, x)
    If all uk's for the user are negative then the user's request is unsatisfied/"unserviced"


    Bike Flow Dynamics:
    Si(t+1) = Si(t) - (Bikes departed during t) + (Bikes arrived during t)

    Overall Goal:
    Optimize policy A to reduce BikeShare congestion given budget constraint B

    MDP Formulation:
    Definition -> (MDP_S, MDP_A, MDP_Pr, MRP_R, MDP_\gamma)
    MDP_S - States
    MDP_A - Actions
    MDP_Pr - Transition Matrix
    MDP_R - Reward
    MDP_\gamma - discount factor (standard RL \gamma) (they use .99)

    MDP_S = st = (S(t), D(t-1), A(t-1), E(t-1), RB(t), U(t))
        S(t) - (defined above) - Supply for each region
        D(t-1) - (defined above) - Demand for all regions at last timestep
        A(t-1) - (defined above) - Arrivals for all regions at last timestep
        E(t-1) - Expense at last timestep
        RB(t) - Remaining budget at timestep t
        U(t) - un-service ratio for each region for some number (not defined in paper) of past timesteps

    MDP_A = at = (p1t, ..., pnt) = price incentive for each region ! Their paper only offer 1 per region to leave from there !
    MDP_R = R(st, at) = ratio of satisfied requests at timestep t
    MDP_Pr = transition probability Pr(s_(t+1) | st,at) = probability st to s_(t+1) under action at

    policy pi_(\theta)(st) = maps current state to an action
    Objective find overall discounted rewards

    Data:
    User and Bike location within a region is drawn from a Uniform distribution
    They specify the initial system supply to be O x (3.65 / 20) where O is the number of orders in their system, however
        they do not specify a time frame and they do distribute bikes based on demand per region for which they do not
        provide data. I believe it is better to allow for varying supply and demand through a set distribution. I will
        generate layout files based on distributions and combinations there of, they will be placed in a directory
        BSS_Inits as .bssEnv files.
    '''

    def __init__(self, systemInitObj, budget, stepFile):
        """
        Initializes the environment

        :param systemInitObj: Object to get system init parameters
        :param budget: budget for the game
        :param stepFile: file to save agent learning progress to
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
            "U": np.zeros((self.systemSize**2,)),
            "E": np.zeros((self.systemSize**2,)),
            "Tdij": systemInitObj.getUsers()
        }

        # Accumulators for performance
        self.accumulatedRew = 0
        self.serviceLevel = 0

        # OpenAi Gym settings
        self.action_space = spaces.Box(0, 5, shape=(self.systemSize**2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, max(100, budget), shape=(6, self.systemSize**2,), dtype=np.float32)

        # Revert to original values
        self.revertToStaticInits()

    def getBudget(self):
        '''
        Getter for the game's budget

        :return: game budget
        '''
        return self.budget

    '''
    Gym top-levels
    '''
    def step(self, action):
        '''
        Take an action to impact the environment and receive feedback

        :param action: [incentive for all 100 regions]

        '''
        # Get user movement matrix for this step
        dij = self.Tdij[self.curStep]

        # Number of users
        numberOfUsers = len(dij)

        # Record number of serviced users
        servicedUsers = 0

        # Record number of times there may be an unservice event
        potentialUnservice = 0

        # Record when an unservice event is fixed
        resolved = 0

        # Iterate through users in the hour
        for user in dij:
            # set users movement interest from i to j
            i = user[0]
            j = user[1]

            # Check if we will need to service users
            if self.S[i] > 0:
                servicedUsers += 1
                self.S[i] -= 1
                self.D[i] += 1
                self.A[j] += 1

            # Need to move the user
            else:
                potentialUnservice += 1

                # Move user
                neighbors = [n for n in self.neighbors(i) if n != -1]
                utilities = [(action[i] - self.walkingCost(), n) for n in neighbors]

                # Find 'best' move
                maxNeighbor = -1
                maxUtil = -1
                for u, n in utilities:
                    if u > maxUtil and self.S[n] > 0:
                        maxUtil = u
                        maxNeighbor = n

                # Check that the move is acceptable
                if maxUtil >= 0 and self.budget >= action[i]:
                    # Service
                    self.D[maxNeighbor] += 1
                    self.A[j] += 1
                    self.S[maxNeighbor] -= 1
                    self.budget -= action[i]
                    self.E[i] += action[i]
                    resolved += 1
                    servicedUsers += 1
                else:
                    # Miss service
                    self.U[i] += 1

        # Handles over head of state transition, users arrive
        self.resolvePreviousState()

        # Format output
        state = self.buildState()

        # Calculate metrics
        serviceLevel = servicedUsers / numberOfUsers
        try:
            reduceUnserviceLevel = 1 - ((potentialUnservice - resolved)/ potentialUnservice)
        except:
            reduceUnserviceLevel = 1.0

        # Check done
        done = self.curStep == self.steps
        info = {}
        self.accumulatedRew += reduceUnserviceLevel
        self.serviceLevel += serviceLevel

        return state, reduceUnserviceLevel, done, info

    def reset(self):
        '''
        Reset environment

        :return: New state
        '''
        self.stepFile.write(str("%.3f,%.3f,%s\n" % (self.accumulatedRew/self.steps, self.serviceLevel/self.steps, str(self.budget))))
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
        print(np.sum(self.S))
        print("State:")
        print(self.buildState())
        print("\n\n\n\n")

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
        state = np.array([
            self.S,
            self.A_1,
            self.D_1,
            self.U_1,
            self.E_1,
            np.full((self.systemSize**2,), self.budget)
        ])
        return state

    def walkingCost(self):
        '''
        Calculate the walking cost from a user to a bike in a neighboring region

        :return: walking cost [0,4)
        '''
        # alpha parameter to adjust budget range
        alpha = 1.0
        userLoc = random.random()
        bikeLoc = 1 + random.random()

        return alpha * (bikeLoc - userLoc)**2

    def resolvePreviousState(self):
        '''
        Resolve previous state by finishing trips and moving
        array counters to the previous timestep
        '''
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

        # Increment step counter
        self.curStep += 1

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

    def neighbors(self, regionI):
        '''
        Outputs neighbors of index regionI

        :param regionI: index of region whose neighbors are to be found
        :return: [up, down, left, right]
        '''
        # Get length of rows, assumes square layout
        rowLen = self.systemSize

        # Calculate index of movement direction
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


