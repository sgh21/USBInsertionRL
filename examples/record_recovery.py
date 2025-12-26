import os
from pynput import keyboard
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
import franka_env.envs.wrappers as wrappers

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful demos to collect.")

def on_press(key):
    try:
        if str(key) == 'Key.space':
            wrappers.is_spacebar_pressed = True
    except AttributeError:
        pass

def on_release(key):
    try:
        if str(key) == 'Key.space':
            wrappers.is_spacebar_pressed = False
    except AttributeError:
        pass

def main(_):
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()


    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    recovery_count = 0
    recovery_needed = FLAGS.successes_needed
    pbar = tqdm(total=recovery_needed)
    trajectory = []
    returns = 0
    
    while recovery_count < recovery_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if transition.get("infos", {}).get("recovery", False):
            transitions.append(copy.deepcopy(transition))
            recovery_count += 1
            pbar.update(1)
        if done:
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./recovery_data"):
        os.makedirs("./recovery_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./recovery_data/{FLAGS.exp_name}_{recovery_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {recovery_needed} recoveries to {file_name}")

if __name__ == "__main__":
    app.run(main)
