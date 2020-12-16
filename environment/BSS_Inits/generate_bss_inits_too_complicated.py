"""
This file was discontinued and remade as generate_bss_inits

Its overcomplicated supply and user interest mechanics made it
difficult for agents to learn

This was left here as a marker of the attempt and for possible use in future projects

author Matthew Schofield
version 11.28.2020
"""

import numpy as np
import pickle as pkl
from environment.BSS_Inits.bss_env_init import BSS_Env_Init
import matplotlib.pyplot as plt

from scipy import signal

'''
Generates .bssEnv files that specify initial

System Size, Steps, Region Supplies and User Demands
'''

def makeGaussianMatrix(systemSize, std):
    """
    Returns a 2D Gaussian Matrix
    """
    # Create vector distributed as a Gaussian distribution
    gaussVector = signal.gaussian(systemSize, std=std).reshape(systemSize, 1)
    # Outer multiply of vector to create a Matrix
    gaussMatrix = np.outer(gaussVector, gaussVector)
    return gaussMatrix

def makeUniformMatrix(systemSize):
    '''
    Build a square matrix to represent a Uniform distribution

    :param systemSize: length of matrix sides
    :return: matrix representing a uniform distribution
    '''
    return np.full((systemSize, systemSize), fill_value=.1)

def normalizeMatrix(systemSize, matrix):
    '''
    Normalize a matrix such that all elements sum to 1.0,
    this allows the matrix to be used as a probability distribution

    :param systemSize: length of matrix sides
    :param matrix: matrix to normalize
    :return: normalized matrix
    '''
    matrixFlattened = matrix.reshape((systemSize ** 2))
    matrixNormalized = np.true_divide(matrixFlattened, sum(matrixFlattened)).reshape((systemSize, systemSize))
    return matrixNormalized

def pickUserDistribution(timeStep, systemSize):
    '''
    Generates User arrival/destination interests based on time of day

    :param timeStep: time of day (hour [0,23])
    :param systemSize: length of square system's sides
    :return: Normalized matrix of user interests
    '''
    # Init uniform noise
    aDistribution = makeUniformMatrix(systemSize)
    dDistribution = makeUniformMatrix(systemSize)
    # midnight to 5am, low random activity
    if timeStep <= 5:
        # The intialized uniform distributions will suffice
        pass
    # Morning rush 7am to 9am
    elif timeStep >= 6 and timeStep <= 9:
        # User arrival activity in residential areas
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        aDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        aDistribution[0:4, 4:8] += upperMiddleDemand

        # User destination intention to business areas
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        dDistribution[5:10, 5:10] += bottomRightIntention

    # Mid-day lull 10am to 3pm, movement between residential and business areas
    elif timeStep >= 10 and timeStep <= 15:
        # User arrivals in both residential and business areas
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        aDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        aDistribution[0:4, 4:8] += upperMiddleDemand
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        aDistribution[5:10, 5:10] += bottomRightIntention

        # User destination intentions in both residential and business areas
        dDistribution = makeUniformMatrix(systemSize)
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        dDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        dDistribution[0:4, 4:8] += upperMiddleDemand
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        dDistribution[5:10, 5:10] += bottomRightIntention

    # Late-Afternoon Rush Hour Home
    elif timeStep >= 16 and timeStep <= 19:
        # User arrivals in business areas
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        aDistribution[5:10, 5:10] += bottomRightIntention

        # User intended destinations in residential areas
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        dDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        dDistribution[0:4, 4:8] += upperMiddleDemand

    # Evening Activity, activity to and from user and leisure areas
    elif timeStep >= 20:
        # User arrivals in both leisure and residential areas
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        aDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        aDistribution[0:4, 4:8] += upperMiddleDemand
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        aDistribution[5:10, 0:5] += bottomRightIntention

        # User intended destinations in both leisure and residential areas
        upperLeftDemand = makeGaussianMatrix(4, 1.5)
        dDistribution[0:4, 0:4] += upperLeftDemand
        upperMiddleDemand = makeGaussianMatrix(4, 1)
        dDistribution[0:4, 4:8] += upperMiddleDemand
        bottomRightIntention = makeGaussianMatrix(5, 1.5)
        dDistribution[5:10, 0:5] += bottomRightIntention
    # Normalize distribution matrices
    return normalizeMatrix(systemSize, aDistribution), normalizeMatrix(systemSize, dDistribution)

def makeSimulatedReplicaOfPaper():
    '''
    Here I attempt to replicate the MoBike Shanghai data shown in the paper

    They mention having 102,361 orders in one month so I interpolate that there are ~ 3,400 orders per day
    '''
    # 5-day
    steps = 24*5

    # 10x10 grid
    systemSize = 10

    # Users
    '''
    Distribution found in a slide in the paper's conference presentation
    '''
    userActivityDistrib = [
        1, .5, .3, .1, .1,
        .5, 2.5, 6, 8.5, 5,
        3.5, 3.5, 4, 4, 4,
        4.5, 5, 9, 10, 9,
        7, 6, 4, 2
    ]

    # Normalize to sum to 1.0
    userActivityDistrib = [u/100 for u in userActivityDistrib]

    # Use probability distribution to generate users per hour
    elementIndices = np.arange(24)
    usersPerHour = np.zeros((24,))
    for u in range(3400):
        usersPerHour[np.random.choice(elementIndices, p=userActivityDistrib)] += 1

    usersActivityOut = []
    regionIndices = np.arange(systemSize ** 2)
    userDemandCountsPerRegion = np.zeros((systemSize**2, ))
    for week in range(5):
        weekActivity = []
        for day in range(5):
            for i, usersInHour in enumerate(usersPerHour):
                hourActivity = []
                arrivalsDistrib, destinationsDistrib = pickUserDistribution(i, systemSize)
                arrivalsDistrib = arrivalsDistrib.reshape((systemSize ** 2,))
                destinationsDistrib = destinationsDistrib.reshape((systemSize ** 2,))
                for user in range(int(usersInHour)):
                    arriv = np.random.choice(regionIndices, p=arrivalsDistrib)
                    userDemandCountsPerRegion[arriv] += 1
                    dept = np.random.choice(regionIndices, p=destinationsDistrib)
                    hourActivity.append([arriv, dept])
                weekActivity.append(hourActivity)
        usersActivityOut.append(np.array(weekActivity))
    users = np.array(usersActivityOut)

    # Supply
    # Concentrate supply in residential and business areas
    supply = np.full((systemSize, systemSize, ), fill_value=2).astype(int)
    supply[0:4, 0:8] += np.full((4,8), fill_value=4)
    supply[5: 10, 0: 5] += np.full((5,5), fill_value=1)
    supply = supply.reshape((systemSize**2,))

    return BSS_Env_Init(steps, systemSize, supply, users)
'''
inits = makeSimulatedReplicaOfPaper()
print(inits.getInitSupply())
pkl.dump(inits, open("paperReplica.pkl", 'wb+'))
'''
'''
Visualizations to confirm
'''

def showDistribution(matrix, title):
    '''
    Show a distribution matrix using pyplot

    :param matrix: matrix to display
    :param title: title of display
    '''
    plt.title(title)
    plt.imshow(matrix, cmap='gray')
    plt.show()

def showUserDistributions():
    '''
    Test function to visualize distributions
    '''
    timeCodes = {
        "Early Morning": 5,
        "Morning Commutes": 8,
        "Afternoon": 13,
        "Evening Commutes": 17,
        "Late Night": 22
    }
    for k in timeCodes.keys():
        aDistribution, dDistribution = pickUserDistribution(timeCodes[k], 10)
        print(aDistribution)
        showDistribution(aDistribution, k + " - User Arrival Pattern")
        print(dDistribution)
        showDistribution(dDistribution, k + " - User Destination Interest Pattern")

def showActivityPerHour():
    '''
    Useful for visualizing total number of users per hours
    '''
    init_users = makeSimulatedReplicaOfPaper().getUsers()[0]
    perHour = np.zeros((24,))
    for i in range(24):
        userArrivals = np.zeros((100,))
        perHour[i] = len(init_users[i])
        for u in init_users[i]:
            userArrivals[u[0]] += 1
    plt.bar(x=np.arange(24), height=perHour)
    plt.show()

showUserDistributions()