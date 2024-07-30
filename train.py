import copy
import json
import os
import random
import time

import numpy as np
import torch
from visdom import Visdom

from case_generator import CaseGenerator
from fjsp_env import FJSPEnv
from plot_gantt import initialize_plt
from schedule_model import Memory, PPO
from validate import get_validate_env, validate


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Tri_VGM(H):
    delta = 1 / H
    w = []
    w1 = 0
    while w1 <= 1:
        w2 = 0
        while w2 + w1 <= 1:
            w3 = 1 - w1 - w2
            w.append([w1, w2, w3])
            w2 += delta
        w1 += delta
    return torch.tensor(w)

def main():
    # PyTorch initialization
    setup_seed(1234)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float)
    else:
        torch.set_default_dtype(torch.float)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = 2 * model_paras["out_size_embed"]
    model_paras["critic_in_dim"] = 2 * model_paras["out_size_embed"]
    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]

    weights = Tri_VGM(16)
    weights = weights[120:]
    for w in range(weights.size(0)):
    # for w in range(1):
        weight = weights[w]
        # weight = torch.tensor([0.5, 0.3, 0.2])
        model_paras["weight"] = weight[-1]
        memories = Memory()
        model = PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
        # 验证集使用默认的IBP参数
        env_valid = get_validate_env(env_valid_paras, weight)
        str_weight = '_'.join([f"{x:.4f}" for x in weight])
        is_viz = train_paras["viz"]
        if is_viz:
            viz = Visdom(env='IBP_{0}'.format(str_weight))

        step_rewards = []
        step_loss = []
        valid_rst = []

        start_time = time.time()
        env = None
        for i in range(1, train_paras["max_iterations"] + 1):
            if (i - 1) % train_paras["parallel_iter"] == 0:
                # 动态生成IBP参数
                s = random.randint(8, 19)
                l = random.randint(2, 4)
                env_paras["performance_period"] = [s, s + l]
                hel = 4 * num_mas  # 功率中位数
                env_paras["HEL"] = hel
                thr = random.randint(40, 60)
                env_paras["THR"] = hel * thr / 100
                nums_ope = [random.randint(3, 5) for _ in range(num_jobs)]
                case = CaseGenerator(num_jobs, num_mas, 3, 5, nums_ope=nums_ope)
                env = FJSPEnv(case=case, env_paras=env_paras, weight=weight)
                print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))
                discount_matrix = torch.zeros((sum(nums_ope), sum(nums_ope)))
                # 填充对角线元素为 1
                discount_matrix.fill_diagonal_(1)
                # 填充对角线右下方元素
                for j in range(1, sum(nums_ope)):
                    discount_matrix[j, :j] = discount_matrix[j - 1, :j] * train_paras["gamma"]

            state = env.state
            done = False
            last_time = time.time()
            while not done:
                with torch.no_grad():
                    actions = model.policy_old(state, memories, flag_sample=True, flag_train=True)
                state, rewards, dones = env.step(actions)
                done = dones.all()
                memories.rewards.append(rewards)

            print("spend_time: ", time.time() - last_time)
            env.reset()

            if i % train_paras["update_timestep"] == 0:
                loss, reward = model.update(memories, train_paras, discount_matrix)
                print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
                step_rewards.append(reward)
                step_loss.append(loss)
                memories.clear_memory()
                if is_viz:
                    viz.line(X=np.array([i]), Y=np.array([reward]),
                             win='window{}'.format(0), update='append', opts=dict(title='reward of training step'))
                    viz.line(X=np.array([i]), Y=np.array([loss]),
                             win='window{}'.format(1), update='append', opts=dict(title='loss of training step'))

            if i % train_paras["save_timestep"] == 0:
                print('\nStart validating')
                vali_result = validate(env_valid, model.policy_old)
                valid_rst.append(vali_result.item())
                if is_viz:
                    viz.line(
                        X=np.array([i]), Y=np.array([vali_result.item()]),
                        win='window{}'.format(2), update='append', opts=dict(title='fitness of valid'))
            if i == train_paras["max_iterations"]:
                os.makedirs('save_1005/{0}'.format(str_weight), exist_ok=True)
                save_file = 'save_1005/{0}.pt'.format(str_weight)
                torch.save(model.policy.state_dict(), save_file)
        print("total_time_{0}: ".format(str_weight), time.time() - start_time)
        np.savetxt('save_1005/{0}/rewards.txt'.format(str_weight), np.array(step_rewards), fmt='%.4f')
        np.savetxt('save_1005/{0}/loss.txt'.format(str_weight), np.array(step_loss), fmt='%.4f')
        np.savetxt('save_1005/{0}/validate.txt'.format(str_weight), np.array(valid_rst), fmt='%.4f')


if __name__ == '__main__':
    main()
