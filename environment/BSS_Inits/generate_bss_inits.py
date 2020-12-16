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
    # Flatten matrix
    matrixFlattened = matrix.reshape((systemSize ** 2))

    # Normalize matrix
    matrixNormalized = np.true_divide(matrixFlattened, sum(matrixFlattened)).reshape((systemSize, systemSize))

    return matrixNormalized

def pickUserDistribution(systemSize):
    '''
    Generates User arrival/destination interests based on time of day

    :param timeStep: time of day (hour [0,23])
    :param systemSize: length of square system's sides
    :return: Normalized matrix of user interests
    '''
    # Init matrices
    aDistribution = makeGaussianMatrix(systemSize, 3)

    dDistribution = makeGaussianMatrix(systemSize, 3)

    # Normalize distribution matrices
    return normalizeMatrix(systemSize, aDistribution), normalizeMatrix(systemSize, dDistribution)

def makeSimulationEnvironment():
    '''
    Here I make a simulation environment
    I attempt to replicate some features of the MoBike Shanghai data shown in the paper

    They mention having 102,361 orders in one month so I interpolate that there are ~ 3,400 orders per day
    Also, I use the users per hour distribution shown in a presentation
    '''
    # 1-day, 24h
    steps = 24

    # 10x10 grid
    systemSize = 10

    # Users
    '''
    Distribution per hour found in a slide in the paper's conference presentation
    
    Users per hour
    '''
    userActivityDistrib = [
        1, .5, .3, .1, .1,
        .5, 2.5, 6, 8.5, 5,
        3.5, 3.5, 4, 4, 4,
        4.5, 5, 9, 10, 9,
        7, 6, 4, 2
    ]

    # Normalize values to sum to 1.0
    userActivityDistrib = [u/100 for u in userActivityDistrib]

    # Use probability distribution to generate users per hour
    # [1,2,...,24], useful for random selection
    elementIndices = np.arange(24)

    # Users per hour counts storage
    usersPerHour = np.zeros((24,))

    # 3,400 users
    for u in range(3400):
        # Randomly select using distribution, while it is simple to instead multiply in 3,400
        # This operation is not expensive and adds slight variation to smaller sets
        # If you want to generate hundreds of thousands(or more) users, just use element-wise multiplication
        hourIndex = np.random.choice(elementIndices, p=userActivityDistrib)
        usersPerHour[hourIndex] += 1

    # Prepare to create final output
    usersActivityOut = []

    # [1,2,..., systemSize], useful for random indexing
    regionIndices = np.arange(systemSize ** 2)

    # Step through
    for i, usersInHour in enumerate(usersPerHour):
        # Will store the users for the hour
        hourActivity = []

        # Get the distribution for the hour somewhat vestigial from an early iteration, but
        # the chain of generation and normalization is useful. No need to change for now.
        arrivalsDistrib, destinationsDistrib = pickUserDistribution(systemSize)

        # Flatten distribution matrices
        arrivalsDistrib = arrivalsDistrib.reshape((systemSize ** 2,))
        destinationsDistrib = destinationsDistrib.reshape((systemSize ** 2,))

        # For number of users in the hour
        for _ in range(int(usersInHour)):
            # Generate arrival and destination interest
            arriv = np.random.choice(regionIndices, p=arrivalsDistrib)
            dest = np.random.choice(regionIndices, p=destinationsDistrib)

            # Save User modeled as pair of arrival and destination interest
            hourActivity.append([arriv, dest])

        # Save the hour
        usersActivityOut.append(hourActivity)

    # Visual check on number of users per hour
    print([len(hour) for hour in usersActivityOut])

    # Save output of users
    users = np.array(usersActivityOut)

    # Supply
    # Concentrate supply in some areas
    '''
    In a systemSize 10x10 = 100
    Supply under these conditions would be 
    100*2 + ((8*4)*4) + (5*5) = 
    200 + 128 + 25 = 
    353
    
    With many peak times requiring between roughly 280 and 340 a supply of 353
    is competitive and allows for effective strategies to show themselves
    '''
    supply = np.full((systemSize, systemSize, ), fill_value=2).astype(int)
    supply[0:4, 0:8] += np.full((4,8), fill_value=4)
    supply[5: 10, 0: 5] += np.full((5,5), fill_value=1)
    supply = supply.reshape((systemSize**2,))

    return BSS_Env_Init(steps, systemSize, supply, users)
'''
# Build BSS Init
inits = makeSimulationEnvironment()

# Visual check
print(inits.getInitSupply())

# Serialize BSS Init object
pkl.dump(inits, open("environment_settings_d.pkl", 'wb+'))
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
    aDistribution, dDistribution = pickUserDistribution(10)
    print(aDistribution)
    showDistribution(aDistribution, "User Arrival Pattern")
    print(dDistribution)
    showDistribution(dDistribution, "User Destination Interest Pattern")

def showActivityPerHour():
    '''
    Useful for visualizing total number of users per hours
    '''
    # Users in init
    init_users = makeSimulationEnvironment().getUsers()

    # Hour counts storage
    perHour = np.zeros((24,))

    # Iterate through hours
    for i in range(24):
        # Init counter for regions
        userArrivals = np.zeros((100,))

        # Count of users per hour
        perHour[i] = len(init_users[i])

        # Count of users per region
        for u in init_users[i]:
            userArrivals[u[0]] += 1

    # Plot users per hour
    plt.bar(x=np.arange(24), height=perHour)
    plt.title("User Arrivals per Hour")
    plt.ylabel("Number of User Arrivals")
    plt.xlabel("Hour [0-23]")
    plt.show()
showActivityPerHour()