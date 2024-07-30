import copy
import math

import torch
from torch import nn
from torch.distributions import Categorical

from actor_critic import MLPActor, MLPCritic, MLPEmded
from embed_ma import EmdedMA
from embed_op import EmdedOP
import torch.nn.functional as F


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []

        self.raw_opes = []
        self.raw_mas = []
        self.raw_edge = []
        self.ope_ma_adj = []
        self.solved_pre = []
        self.ope_pre_adj = []
        self.jobs_gather = []
        self.eligible = []
        self.norm_glo = []
        self.action_indexes = []

    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.raw_edge[:]
        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.solved_pre[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.norm_glo[:]
        del self.action_indexes[:]


class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.in_size_edge = model_paras["in_size_edge"]
        self.in_size_global = model_paras["in_size_global"]
        self.out_size_embed = model_paras["out_size_embed"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.hidden_size_glo = model_paras["hidden_size_glo"]
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.embed_layer = model_paras["embed_layer"]
        self.weight = model_paras["weight"]

        self.get_opes = nn.ModuleList()
        self.get_opes.append(
            EmdedOP(self.in_size_ope, self.in_size_ma, self.in_size_edge, self.out_size_embed, self.hidden_size_ope))
        for i in range(1, self.embed_layer):
            self.get_opes.append(
                EmdedOP(self.out_size_embed, self.out_size_embed, self.in_size_edge, self.out_size_embed,
                        self.hidden_size_ope))
        self.get_mas = nn.ModuleList()
        self.get_mas.append(EmdedMA(self.out_size_embed, self.in_size_ma, self.in_size_edge, self.out_size_embed))
        for i in range(1, self.embed_layer):
            self.get_mas.append(EmdedMA(self.out_size_embed, self.out_size_embed, self.in_size_edge,
                                        self.out_size_embed))
        self.glo_embed = MLPEmded(self.in_size_global, self.hidden_size_glo, 2 * self.out_size_embed)
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, 1)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1)

    def forward(self, state, memories, flag_sample=False, flag_train=False):
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        raw_edge = state.feat_edge_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, raw_edge, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_edge = (copy.deepcopy(features[2]))
        op_adj_in = state.ope_pre_adj_batch[batch_idxes]
        ma_adj_in = state.solved_pre[batch_idxes]
        op_ma_adj = state.ope_ma_adj_batch[batch_idxes]
        # L iterations of the HGNN
        for i in range(self.embed_layer):
            # First Stage, operation node embedding
            # shape: [len(batch_idxes), num_ops, out_size_ope]
            h_ops = self.get_opes[i](features[0], features[1], features[2], op_adj_in, ma_adj_in, op_ma_adj)
            features = (h_ops, features[1], features[2])
            # Second Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_mas[i](features[0], features[1], features[2], op_ma_adj)
            features = (features[0], h_mas, features[2])
        # Stacking and pooling
        h_mas_pooled = features[1].mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(features[0][i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)
        else:
            h_opes_pooled = features[0].mean(dim=-2)
        # shape:[len(batch_idxes), 2*out_size_embed]
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, self.out_size_embed)[batch_idxes]
        # shape:[len(batch_idxes),num_jobs, out_size_embed]
        h_jobs = features[0].gather(1, jobs_gather)
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = op_ma_adj.gather(1, ope_step_batch[..., :, None].expand(-1, -1, raw_mas.size(1))[
            batch_idxes])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(eligible_proc)
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & (eligible_proc == 1)
        # shape: [len(batch_idxes), num_jobs, num_mas,out_size_embed]
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, raw_mas.size(1), -1)
        h_mas_padding = features[1].unsqueeze(1).expand_as(h_jobs_padding)
        # shape: [len(batch_idxes), num_jobs, num_mas,2*out_size_embed]
        h_padding = torch.cat((h_jobs_padding, h_mas_padding), dim=-1)
        h_pooled_padding = h_pooled[:, None, None, :].expand_as(h_padding)
        # 全局特征嵌入
        norm_glo = state.feat_glo_batch[batch_idxes]
        h_glo = self.glo_embed(norm_glo)
        h_glo_padding = h_glo.unsqueeze(1).unsqueeze(1).expand_as(h_padding)
        h_actions = h_padding + h_glo_padding - h_pooled_padding

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        # Store data in memory during training
        if flag_train == True:
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.raw_edge.append(copy.deepcopy(norm_edge))
            memories.ope_ma_adj.append(copy.deepcopy(op_ma_adj))
            memories.ope_pre_adj.append(copy.deepcopy(op_adj_in))
            memories.solved_pre.append(copy.deepcopy(ma_adj_in))
            memories.norm_glo.append(copy.deepcopy(norm_glo))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        jobs = (action_indexes / raw_mas.size(1)).long()
        mas = (action_indexes % raw_mas.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, raw_opes, raw_mas, raw_edge, op_adj_in, ma_adj_in, op_ma_adj, jobs_gather, norm_glo, eligible,
                 action_envs):
        features = (raw_opes, raw_mas, raw_edge)
        # L iterations of the HGNN
        for i in range(self.embed_layer):
            h_ops = self.get_opes[i](features[0], features[1], features[2], op_adj_in, ma_adj_in, op_ma_adj)
            features = (h_ops, features[1], features[2])
            h_mas = self.get_mas[i](features[0], features[1], features[2], op_ma_adj)
            features = (features[0], h_mas, features[2])
        h_opes_pooled = features[0].mean(dim=-2)
        h_mas_pooled = features[1].mean(dim=-2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        h_jobs = features[0].gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, raw_mas.size(1), -1)
        h_mas_padding = features[1].unsqueeze(1).expand_as(h_jobs_padding)
        # shape: [len(batch_idxes), num_jobs, num_mas,2*out_size_embed]
        h_padding = torch.cat((h_jobs_padding, h_mas_padding), dim=-1)
        h_pooled_padding = h_pooled[:, None, None, :].expand_as(h_padding)
        h_glo = self.glo_embed(norm_glo)
        h_glo_padding = h_glo.unsqueeze(1).unsqueeze(1).expand_as(h_padding)
        h_actions = h_padding + h_glo_padding - h_pooled_padding
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        h_critic = h_pooled + h_glo
        state_values = self.critic(h_critic)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze(), dist_entropys

    def get_normalized(self, raw_opes, raw_mas, raw_edge, batch_idxes, nums_opes, flag_sample=False, flag_train=False):

        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            edge_copy = copy.deepcopy(raw_edge)
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                for j in range(raw_edge.size(-1)):
                    proc_values = raw_edge[i, :nums_opes[i], :, j]
                    proc_norm = self.feature_normalize(proc_values)
                    edge_copy[i, :nums_opes[i], :, j] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            edge_norm = edge_copy
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            sliced_tensors = torch.unbind(raw_edge, dim=-1)
            normalized_tensors = []
            for t in sliced_tensors:
                normalized_tensors.append(self.feature_normalize(t))
            stacked_tensor = torch.stack(normalized_tensors, dim=-1)
            edge_norm = stacked_tensor  # shape: [len(batch_idxes), num_ops, num_mas, in_size_edge]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                edge_norm)

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))


class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)

        self.MseLoss = nn.MSELoss()

    def update(self, memory, train_paras, discount_matrix):
        minibatch_size = train_paras["minibatch_size"]
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_edge = torch.stack(memory.raw_edge, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_solved_pre = torch.stack(memory.solved_pre, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        old_norm_glo = torch.stack(memory.norm_glo, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1)

        discounted_reward = torch.matmul(memory_rewards, discount_matrix)
        rewards = torch.sum(discounted_reward[:, 0]) / self.num_envs
        # 归一化奖励
        mean_reward = torch.mean(discounted_reward, dim=-1, keepdim=True)
        std_reward = torch.std(discounted_reward, dim=-1, keepdim=True)
        discounted_reward = ((discounted_reward - mean_reward) / (std_reward + 1e-5)).view(-1)

        loss_epochs = 0
        full_batch_size = old_raw_opes.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    old_raw_opes[start_idx: end_idx, :, :],
                    old_raw_mas[start_idx: end_idx, :, :],
                    old_raw_edge[start_idx: end_idx, :, :, :],
                    old_ope_pre_adj[start_idx: end_idx, :, :],
                    old_solved_pre[start_idx: end_idx, :, :],
                    old_ope_ma_adj[start_idx: end_idx, :, :],
                    old_jobs_gather[start_idx: end_idx, :, :],
                    old_norm_glo[start_idx: end_idx, :],
                    old_eligible[start_idx: end_idx, :, :],
                    old_action_envs[start_idx: end_idx])
                ratios = torch.exp(logprobs - old_logprobs[i * minibatch_size:(i + 1) * minibatch_size].detach())
                advantages = discounted_reward[i * minibatch_size:(i + 1) * minibatch_size] - state_values.detach()
                advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-5)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = (- self.A_coeff * torch.min(surr1, surr2)
                        + self.vf_coeff * self.MseLoss(state_values,
                                                       discounted_reward[i * minibatch_size:(i + 1) * minibatch_size])
                        - self.entropy_coeff * dist_entropy)
                loss_epochs += loss.mean().detach()
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_epochs.item() / self.K_epochs, rewards.item()
