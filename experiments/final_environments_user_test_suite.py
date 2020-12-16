import sys

# Important for cloud compute
# Change this to the directory that contains the environments directory
sys.path.insert(0, "/home/schofielm0")

# Imports
from environment.bss_controller_base import BSS_Controller_Base
from environment.bss_controller_base_user import BSS_Controller_User
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
for letter in ["a", "b", "c" "d"]:
    env_settings_init = pkl.load(open("../environment/BSS_Inits/environment_settings_"+letter+".pkl", 'rb'))
    accumulatedRew = 0
    iterations = 0
    learnSteps = 1200000
    evaluationLen = 10000
    for budget in [500, 1000, 2000, 10000]:
        for env, expName in [
            (
                BSS_Controller_User(env_settings_init, budget, open(letter + "/v5_stepsBudget" + str(budget) + ".csv", 'a+')),
                "v5"
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



