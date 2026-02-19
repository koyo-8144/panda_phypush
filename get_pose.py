import logging
import time
import numpy as np
import panda_py
from panda_py import libfranka, constants
from matplotlib import pyplot as plt

# 1. Configuration
hostname = '172.22.2.3'
username = 'cobotmakerspace'
password = 'cobotmakerspace'

# Activate information log level as panda-py is chatty
logging.basicConfig(level=logging.INFO)

def run_panda_test():
    try:
        # 2. Desk Connection (Code Block 1)
        # Unlock brakes and activate FCI for control
        print("\n--- Connecting to Desk ---")
        desk = panda_py.Desk(hostname, username, password)
        # desk.unlock()
        # desk.activate_fci()

        # 3. Robot and Gripper Initialization (Code Block 2)
        print("\n--- Initializing Robot and Gripper ---")
        panda = panda_py.Panda(hostname)
        gripper = libfranka.Gripper(hostname)


        pose = panda.get_pose()
        print("pose: \n", pose)
        print("q: \n", panda.get_state().q)
        # print(f"Current State: {panda.get_state()}")
        # print(f"Model Info: {panda.get_model()}")



    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_panda_test()