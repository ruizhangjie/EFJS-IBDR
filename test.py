import copy
import json
import os
import time

import numpy as np
import torch

from fjsp_env import FJSPEnv
from plot_gantt import initialize_plt
from schedule_model import Memory, PPO


def main():
    # PyTorch initialization
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float)
    else:
        torch.set_default_dtype(torch.float)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)

    # Load config and init objects
    with open("config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    train_paras = load_dict["train_paras"]
    model_paras = load_dict["model_paras"]
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = 2 * model_paras["out_size_embed"]
    model_paras["critic_in_dim"] = 2 * model_paras["out_size_embed"]
    env_test_paras["performance_period"] = [10, 15]
    env_test_paras["HEL"] = 24
    env_test_paras["THR"] = 20
    mod_files = os.listdir('model/')[:]

    start = time.time()
    for gen in range(1):
        fitness = []
        for i_mod in range(len(mod_files)):
            mod = mod_files[i_mod]
            numbers_str = mod[:-3].split("_")
            numbers_float = [float(num) for num in numbers_str]
            weight = torch.tensor(numbers_float)
            model_paras["weight"] = weight[-1]
            memories = Memory()
            model = PPO(model_paras, train_paras)
            if device.type == 'cuda':
                model_CKPT = torch.load('model/' + mod)
            else:
                model_CKPT = torch.load('model/' + mod, map_location='cpu')
            print('\nloading checkpoint:', mod)
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)

            step_time_last = time.time()

            case = os.listdir('data_test/1_Brandimarte/')
            file = 'data_test/1_Brandimarte/' + case[1]
            # case = os.listdir('test/')
            # case[0] = 'test/' + case[0]
            env = FJSPEnv(case=file, env_paras=env_test_paras, weight=weight, data_source='file')
            objs = schedule(env, model, memories)
            fitness.append(objs.squeeze())
            print("rule_spend_time: ", time.time() - step_time_last)
        print("total_spend_time: ", time.time() - start)
        pop = torch.stack(fitness, dim=0)
        pop = pop.clone().cpu().numpy()
        all_fit = np.unique(pop, axis=0)
        real_pop = find_non_dominated_solutions(all_fit)
        np.savetxt('test_result/MK1_DRL_{0}.txt'.format(gen), real_pop, fmt='%.2f', delimiter=' ')



def schedule(env, model, memories):
    # Get state and completion signal
    state = env.state
    done = False  # Unfinished at the beginning
    while not done:
        with torch.no_grad():
            actions = model.policy_old(state, memories, flag_sample=True, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    # f = copy.deepcopy(env.f).squeeze()
    # m = float(f[0])
    # t = float(f[1])
    # i = float(f[2])
    # path = 'test_result/{0}_{1}_{2}.jpg'.format(round(m, 2), round(t, 2), round(i, 2))
    # ma_start = env.gantt_batch[..., 0].squeeze()
    # ma_end = env.gantt_batch[..., 1].squeeze()
    # ma_op = env.gantt_batch[..., 2].squeeze()
    # initialize_plt(env.num_mas, env.num_jobs, ma_op, ma_start,
    #                ma_end, env.num_ope_biases_batch.squeeze(), path)
    return copy.deepcopy(env.f)


def find_non_dominated_solutions(fitness_values):
    # 获取个体数量和目标数量
    pop_size, num_objectives = fitness_values.shape

    # 标记所有个体是否是非支配解，初始全为True
    is_non_dominated = np.ones(pop_size, dtype=bool)

    for i in range(pop_size):
        for j in range(pop_size):
            if i != j:
                # 检查个体j是否支配个体i
                if np.all(fitness_values[j] <= fitness_values[i]) and np.any(fitness_values[j] < fitness_values[i]):
                    is_non_dominated[i] = False
                    break

    # 返回非支配解
    return fitness_values[is_non_dominated]


if __name__ == '__main__':
    main()
