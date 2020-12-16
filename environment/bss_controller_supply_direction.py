'''
v4
'''


import gym
from gym import spaces
import numpy as np
from environment.bss_controller_base_direction import BSS_Controller_Base_Direction

class BSS_Controller_Supply_Direction(BSS_Controller_Base_Direction):
    '''
    Author Matthew Schofield
    Version 11.16.2020

    Changes from paper:

    Reduced state space to only supply
    '''

    def __init__(self, systemInitObj, budget, stepFile):
        """
        Initializes the environment
        """
        super().__init__(systemInitObj, budget, stepFile)

        # OpenAi Gym settings
        self.action_space = spaces.Box(0, 5, shape=(self.systemSize**2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 100.0, shape=(self.systemSize**2,), dtype=np.float32)

    '''
    Helper methods
    '''
    def buildState(self):
        state = self.S
        return state