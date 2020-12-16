"""
v3

author Matthew Schofield
"""

from environment.bss_controller_base import BSS_Controller_Base

class BSS_Controller_Base_Direction(BSS_Controller_Base):
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

    '''
    Helper methods
    '''
    def step(self, action):
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
            # Number of user moving from i to j right now
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
                utilities = [(action[n] - self.walkingCost(), n) for n in neighbors]

                # Find 'best' move
                maxNeighbor = -1
                maxUtil = -1
                for u, n in utilities:
                    if u > maxUtil and self.S[n] > 0:
                        maxUtil = u
                        maxNeighbor = n

                # Check that the move is acceptable
                if maxUtil >= 0 and self.budget >= action[maxNeighbor]:
                    # Service
                    self.D[maxNeighbor] += 1
                    self.A[j] += 1
                    self.S[maxNeighbor] -= 1
                    self.budget -= action[maxNeighbor]
                    self.E[i] += action[maxNeighbor]
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
