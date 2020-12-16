import sys

# Important for cloud compute
# Change this to the directory that contains the environments directory
sys.path.insert(0, "/home/schofielm0")

# Imports
from environment.bss_controller_base import BSS_Controller_Base
from environment.bss_controller_supply_state import BSS_Controller_Base_Supply
from environment.bss_controller_base_direction import BSS_Controller_Base_Direction
from environment.bss_controller_supply_direction import BSS_Controller_Supply_Direction
from environment.bss_controller_supply_direction_prediction import BSS_Controller_Supply_Direction_Prediction
import numpy as np
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
import time
import pickle as pkl

'''
Experimental Set-Up
'''
# Step through experiment settings codes
# Letter Arrival  Destination
# a      uniform  uniform
# b      gaussian uniform
# c      uniform  gaussian
# d      gaussian gaussian
for letter in ["a", "b", "c", "d"]:
    env_settings_init = pkl.load(open("../environment/BSS_Inits/environment_settings_"+letter+".pkl", 'rb'))
    accumulatedRew = 0
    iterations = 0
    learnSteps = 800000
    evaluationLen = 2400
    for budget in [500, 1000, 2000, 10000]:
        for env, expName in [
            (
                BSS_Controller_Base(env_settings_init, budget, open(letter + "/v1_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v1"
            ),
            (
                BSS_Controller_Base_Supply(env_settings_init, budget, open(letter + "/v2_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v2"
            ),
            (
                BSS_Controller_Base_Direction(env_settings_init, budget, open(letter + "/v3_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v3"
            ),
            (
                BSS_Controller_Supply_Direction(env_settings_init, budget, open(letter+ "/v4_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v4"
            ),
            (
                BSS_Controller_Supply_Direction_Prediction(env_settings_init, budget, open(letter+ "/v6_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v6"
            )
        ]:
            accumulatedRew = 0
            iterations = 0
            outFile = open(letter + "/" + expName + "_perfBudget" + str(budget) + ".csv", 'a+')
            agent = TRPO(MlpPolicy, env)
            state = env.reset()
            start = time.time()
            print("Beginning to learn " + expName)
            agent.learn(learnSteps)
            print(time.time() - start)
            print("\tDone Learning")
            for _ in range(evaluationLen):
                action = agent.predict(state)
                state, reward, done, info = env.step(action[0])
                accumulatedRew += reward
                iterations += 1
                if done:
                    outFile.write(str("%.4f" % (accumulatedRew/iterations)) + "," + str(env.getBudget()) + "\n")
                    accumulatedRew = 0
                    iterations = 0
                    env.reset()
            outFile.close()
            env.close()

    '''
    No Agent
    '''
    print("No agent")
    budget = 0
    env = BSS_Controller_Base(env_settings_init, budget, open(letter + "/noAgent_steps.csv", 'a+'))
    env.reset()
    noAgent = open("noAgent.csv", "a+")
    env.reset()
    for _ in range(evaluationLen):
        state, reward, done, info = env.step(np.zeros((100,))) # take a random action
        print(reward)
        accumulatedRew += reward
        iterations += 1
        if done:
            noAgent.write(str("%.4f" % (accumulatedRew/iterations)) + "," + str(env.getBudget()) + "\n")
            accumulatedRew = 0
            iterations = 0
            env.reset()
    noAgent.close()
    env.close()


    '''
    EmpOpt Agent
    '''
    print("Opt agent")
    accumulatedRew = 0
    iterations = 0
    env = BSS_Controller_Base(env_settings_init, 99999999, open(letter+"/opt_steps.csv", 'a+'))
    env.reset()
    noAgent = open(letter + "/opt.csv", "a+")
    env.reset()
    for _ in range(evaluationLen):
        state, reward, done, info = env.step(np.full((100,), 4.0)) # take a random action

        accumulatedRew += reward
        iterations += 1
        if done:
            noAgent.write(str("%.4f" % (accumulatedRew/iterations)) + "," + str(env.getBudget()) + "\n")
            accumulatedRew = 0
            iterations = 0
            env.reset()
    noAgent.close()
    env.close()
