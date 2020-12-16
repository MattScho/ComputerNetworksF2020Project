'''
Data Structure to define BSS initializations

Schema

stepsPerEpisode = int (24)
systemSize = 10
supply = [r1, r2, ..., r3] len 100
users = [
            [ [a1, d1], [a2, d2], ..., [an, dn]],
            [],
            ...,
            []
        ]
'''

class BSS_Env_Init:

    def __init__(self, stepsPerEpisode, systemSize, supply, users):
        # Steps per episode/game, aka number of hours
        self.stepsPerEpisode = stepsPerEpisode
        self.systemSize = systemSize
        self.initSupplies = supply
        self.users = users

    def getStepsPerEpisode(self):
        return self.stepsPerEpisode

    def getInitSupply(self):
        return self.initSupplies

    def getUsers(self):
        return self.users

    def getSystemSize(self):
        return self.systemSize