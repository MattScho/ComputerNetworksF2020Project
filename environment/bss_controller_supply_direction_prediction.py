'''
v6
'''


import gym
from gym import spaces
import numpy as np
from environment.bss_controller_base_direction import BSS_Controller_Base_Direction

class BSS_Controller_Supply_Direction_Prediction(BSS_Controller_Base_Direction):
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
        self.observation_space = spaces.Box(0.0, 100.0, shape=(3, self.systemSize**2,), dtype=np.float32)

        # Calculate prediction maps
        self.outgoing_prediction = {}
        self.incoming_prediction = {}

        self.build_predictions()

    def build_predictions(self):
        incoming = np.zeros(self.systemSize**2,)
        for i in range(24):
            self.incoming_prediction[i] = incoming
            outgoing = np.zeros(self.systemSize**2,)
            incoming = np.zeros(self.systemSize**2,)
            for user in self.Tdij[i]:
                outgoing[user[0]] += 1
                incoming[user[1]] += 1
            self.outgoing_prediction[i] = outgoing

        self.outgoing_prediction[24] = np.zeros(self.systemSize**2,)
        self.incoming_prediction[24] = np.zeros(self.systemSize**2,)

    '''
    Helper methods
    '''
    def buildState(self):
        state = self.S
        state = np.array([state, self.outgoing_prediction[self.curStep], self.incoming_prediction[self.curStep]])

        return state