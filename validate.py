import copy
import os
import time

import torch

from fjsp_env import FJSPEnv
from plot_gantt import initialize_plt
from schedule_model import Memory


def get_validate_env(env_paras, weight):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "data_dev/{0}{1}/".format(str.zfill(str(env_paras["num_jobs"]), 2),
                                          str.zfill(str(env_paras["num_mas"]), 2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path + valid_data_files[i]
    env = FJSPEnv(case=valid_data_files, env_paras=env_paras, weight=weight, data_source='file')
    return env


def validate(env, model):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    memory = Memory()
    state = env.state
    done = False
    while not done:
        with torch.no_grad():
            actions = model(state, memory)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    fit = copy.deepcopy(env.fitness.mean())
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return fit
