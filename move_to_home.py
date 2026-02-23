import logging
import time
import numpy as np
import panda_py
from panda_py import libfranka, constants
from matplotlib import pyplot as plt
from const import HOSTNAME

# 1. Configuration
username = 'cobotmakerspace'
password = 'cobotmakerspace'


def move_to_pushset():
    logging.basicConfig(level=logging.INFO)
    
    try:
        desk = panda_py.Desk(HOSTNAME, username, password)
        # desk.unlock()
        # desk.activate_fci()

        # 1. Connect to Robot
        print(f"Connecting to {HOSTNAME}...")
        panda = panda_py.Panda(HOSTNAME)
        gripper = libfranka.Gripper(HOSTNAME)
        
        # 2. Reset to Neutral first
        # This clears any previous weird states and gives us a clean starting point.
        print("Moving to Neutral (Start) Position...")
        panda.move_to_start()
        

    except Exception as e:
        print(f"An error occurred: {e}")
        # If it fails due to limits, you might need to adjust PUSHSET_Q slightly.

if __name__ == "__main__":
    move_to_pushset()