import copy
from dataclasses import dataclass

import torch

from load_data import load_fjs, nums_detec
from refer_point import Refer


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    feat_glo_batch: torch.Tensor = None
    feat_edge_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None
    solved_pre: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, feat_glo_batch, feat_edge_batch, ope_ma_adj_batch,
               mask_job_finish_batch, ope_step_batch, solved_pre):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_glo_batch = feat_glo_batch
        self.feat_edge_batch = feat_edge_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.ope_step_batch = ope_step_batch
        self.solved_pre = solved_pre


def load_benchmark(path):
    with open(path) as file_object:
        time = []
        power = []
        lines = file_object.readlines()
        flag = True
        for j in range(len(lines)):
            if lines[j] == '##==##\n':
                power.append(lines[j])
                flag = False
            else:
                if flag:
                    time.append(lines[j])
                else:
                    power.append(lines[j])
    return time, power


def convert_feat(feat_job_batch, opes_appertain_batch):
    return feat_job_batch.gather(1, opes_appertain_batch)


def count_intersections(start, end, slot):
    # 计算性能期OP数量
    # 扩展维度以便进行广播
    start_expanded = start.unsqueeze(-1)
    end_expanded = end.unsqueeze(-1)

    # 创建时间段范围
    time_range = slot.unsqueeze(0).unsqueeze(0)

    # 创建掩码，判断时间段范围是否在start和end之间
    mask = (time_range >= start_expanded) & (time_range <= end_expanded - 1)

    # 计算交集掩码
    intersection_mask = mask

    # 统计每个时间段的交集数量
    intersection_count = intersection_mask.any(dim=-1).sum(dim=-1)

    return intersection_count


def process_tensor(a, b, c):
    # 计算有效次数和平均功率
    mask = a <= b
    counts = mask.sum(dim=1)
    # 使用masked_fill将不满足条件的元素设置为0，然后直接计算平均值
    # 先用 mask 把 a 中不符合条件的元素替换为 0
    masked_a = a.masked_fill(~mask, 0)
    # 计算总和，然后除以非零元素的个数来得到平均值
    sum_values = masked_a.sum(dim=1)
    # 避免除以0，使用 clamp 方法确保至少为1
    averages = sum_values / counts.clamp(min=1)
    # 对于没有任何符合条件的行，将平均值设置为0
    averages[counts == 0] = c
    return counts, averages


def calculate_power_over_time(a, b, c, d):
    c_expanded = c.unsqueeze(0).unsqueeze(0)
    # 计算每个时间点是否在区间内
    in_range = (c_expanded >= a.unsqueeze(-1)) & (c_expanded <= b.unsqueeze(-1) - 1)
    # 使用广播将d扩展至与c_expanded相同的维度
    d_expanded = d.unsqueeze(-1).expand(-1, -1, len(c))
    # 仅将位于时间区间内的功率值累加
    total_power = torch.where(in_range, d_expanded, torch.zeros_like(d_expanded)).sum(dim=1)
    return total_power


class FJSPEnv:
    '''
    FJSP environment
    '''

    def __init__(self, case, env_paras, weight, data_source='case'):

        # load paras
        # static
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.w = weight  # 权重
        self.performance_period = torch.arange(env_paras["performance_period"][0] * 4,
                                               env_paras["performance_period"][1] * 4)
        self.hel = env_paras["HEL"]
        self.thr = env_paras["THR"]
        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        lines_power = []
        spower = []
        if data_source == 'case':  # Generate instances through generators
            for i in range(self.batch_size):
                t, p = case.get_case(i)
                lines.append(t)
                lines_power.append(p)
                self.num_jobs, self.num_mas, num_opes = nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
        else:  # Load instances from files
            for i in range(self.batch_size):
                t, p = load_benchmark(case)
                lines.append(t)
                lines_power.append(p)
                self.num_jobs, self.num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        # load feats
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], lines_power[i], self.num_mas, self.num_opes, 36)
            # standby_power_ma = torch.tensor([0.2, 0.1, 0.3, 0.2, 0.1])
            standby_power_ma = torch.tensor([0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1,0.2, 0.1, 0.3, 0.2, 0.1])

            spower.append(standby_power_ma)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0)
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[6], dim=0).float()  # 方便点乘
        self.proc_powers_batch = torch.stack(tensors[7], dim=0)

        # static feats
        # shape: (batch_size, num_opes, num_opes), 工艺顺序约束
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)  # 入邻居邻接矩阵
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[3], dim=0).long()  # 指定类型方便索引
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[5], dim=0)
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        self.standby_powers_batch = torch.stack(spower, dim=0)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.N = torch.zeros(self.batch_size)  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
             ope:
                0-调度标识
                1-邻居机器节点数量
                2-工件剩余工序数量
                3-开始时间
                4-完工时间 
                5-工件剩余工作量
                6-加工能耗        
             ma:
                0-邻居工序节点数量
                1-释放时间
                2-待机时长
                3-待机能耗
             glo:
                0-加工工序数量
                1-成功激励次数
                2-有效平均功率
             弧：
                0-加工时间
                1-加工功率   
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = convert_feat(self.nums_ope_batch, self.opes_appertain_batch)
        pt_copy = copy.deepcopy(self.proc_times_batch)
        mask = torch.all(pt_copy == 0, dim=-1)  # 指示用来填充的op
        pt_copy[mask] = 1
        pt_copy[pt_copy == 0] = float('inf')
        self.estimate_pt = torch.min(pt_copy, dim=-1)[0]
        feat_opes_batch[:, 3, :] = torch.bmm(self.estimate_pt.unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        feat_opes_batch[:, 4, :] = feat_opes_batch[:, 3, :] + self.estimate_pt
        t0 = convert_feat(feat_opes_batch[:, 4, :], self.end_ope_biases_batch)
        t1 = convert_feat(feat_opes_batch[:, 3, :], self.ope_step_batch)
        feat_opes_batch[:, 5, :] = convert_feat(t0 - t1, self.opes_appertain_batch)
        pp_copy = copy.deepcopy(self.proc_powers_batch)
        pp_copy[mask] = 1
        pp_copy[pp_copy == 0] = float('inf')
        self.estimate_pp = torch.min(pp_copy, dim=-1)[0]
        feat_opes_batch[:, 6, :] = self.estimate_pp * self.estimate_pt
        # 填充的op部分特征赋值为0
        feat_opes_batch[mask.unsqueeze(1).expand(-1, self.paras["ope_feat_dim"], -1)] = 0
        feat_glo_batch = torch.zeros(size=(self.batch_size, self.paras["glo_feat_dim"]))
        feat_glo_batch[:, 0] = count_intersections(feat_opes_batch[:, 3, :], feat_opes_batch[:, 4, :],
                                                   self.performance_period)
        demand_per = calculate_power_over_time(feat_opes_batch[:, 3, :], feat_opes_batch[:, 4, :],
                                               self.performance_period, self.estimate_pp)
        feat_glo_batch[:, 1], feat_glo_batch[:, 2] = process_tensor(demand_per, self.hel - self.thr, self.hel)
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_glo_batch = feat_glo_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_edge_batch = torch.stack((self.proc_times_batch, self.proc_powers_batch), dim=-1)
        # feat_glo最小最大值归一化
        self.feat_glo_batch[:, 0] = self.feat_glo_batch[:, 0] / self.nums_opes
        self.feat_glo_batch[:, 1] = self.feat_glo_batch[:, 1] / len(self.performance_period)
        self.feat_glo_batch[:, 2] = self.feat_glo_batch[:, 2] / (self.hel - self.thr)

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        '''
          Partial Schedule (state), dynamic
              0-开始加工时间
              1-结束加工时间
              2-工序Idx
              3-加工功率
              4-待机开始时间
              5-待机结束时间
        '''
        self.gantt_batch = -1 * torch.ones(size=(self.batch_size, self.num_mas, self.num_opes, 6))
        # 同一机器上的工序队列
        self.solved_pre = torch.full(size=(self.batch_size, self.num_opes, self.num_opes), dtype=torch.bool,
                                     fill_value=False)
        # 预计适应度
        makespan = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        tpc = torch.sum(self.feat_opes_batch[:, 6, :], dim=1)
        deal_demand = torch.where(demand_per <= self.hel - self.thr, demand_per, self.hel)
        ir = torch.sum(deal_demand, dim=-1)
        self.f = torch.stack([makespan, tpc, ir], dim=-1)
        # self.Z = torch.zeros((self.batch_size, 3))
        # for i in range(self.batch_size):
        #     sb_power = self.standby_powers_batch[i].unsqueeze(-1).expand(-1, self.proc_times_batch.shape[1])
        #     refer = Refer(tensors[0][i], tensors[7][i], tensors[1][i], tensors[4][i], tensors[5][i],
        #                   self.performance_period, sb_power, self.fsl - self.thr)
        #     a, b, c = refer.get_z()
        #     self.Z[i, :] = torch.tensor([a, b, c])
        # self.fitness = torch.sum(self.w * (f / self.Z), dim=-1)
        self.fitness = torch.sum(self.w * self.f, dim=-1)

        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch,
                              feat_mas_batch=self.feat_mas_batch,
                              feat_glo_batch=self.feat_glo_batch,
                              feat_edge_batch=self.feat_edge_batch,
                              ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              nums_opes_batch=self.nums_opes,
                              solved_pre=self.solved_pre)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_feat_glo_batch = copy.deepcopy(self.feat_glo_batch)
        self.old_feat_edge_batch = copy.deepcopy(self.feat_edge_batch)
        self.old_state = copy.deepcopy(self.state)
        self.old_proc_powers_batch = copy.deepcopy(self.proc_powers_batch)
        self.old_fitness = copy.deepcopy(self.fitness)
        self.old_estimate_pt = copy.deepcopy(self.estimate_pt)
        self.old_estimate_pp = copy.deepcopy(self.estimate_pp)

    def step(self, actions):
        '''
        Environment transition function
        '''
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch
        self.proc_powers_batch *= self.ope_ma_adj_batch

        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        cols = torch.arange(self.num_opes).expand(start_ope.size(0), self.num_opes)
        mask0 = (cols >= start_ope.unsqueeze(-1)) & (cols <= end_ope.unsqueeze(-1))
        value = self.feat_opes_batch[self.batch_idxes, 2, :]
        value[mask0] -= 1
        self.feat_opes_batch[self.batch_idxes, :2, opes] = 1
        self.feat_opes_batch[self.batch_idxes, 2, :] = value
        # Update 'Start time' and 'Job completion time'
        job_rdy_time = self.feat_opes_batch[self.batch_idxes, 3, opes]
        ma_rdy_time = self.feat_mas_batch[self.batch_idxes, 1, mas]
        start_act = torch.max(ma_rdy_time, job_rdy_time)
        end_act = start_act + proc_times
        self.feat_opes_batch[self.batch_idxes, 3, opes] = start_act
        self.feat_opes_batch[self.batch_idxes, 4, opes] = end_act
        # 更新已调度工序预计加工时间
        self.estimate_pt[self.batch_idxes, opes] = proc_times
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 3, :] * is_scheduled
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        # 已调度工序为实际结束时间，未调度为预计加工时间
        estimate_times = torch.bmm((start_times + self.estimate_pt[self.batch_idxes, :]).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled
        self.feat_opes_batch[self.batch_idxes, 3, :] = start_times + estimate_times
        self.feat_opes_batch[self.batch_idxes, 4, :] = self.feat_opes_batch[self.batch_idxes, 3, :] + self.estimate_pt[
                                                                                                      self.batch_idxes,
                                                                                                      :]
        t0 = convert_feat(self.feat_opes_batch[self.batch_idxes, 4, :], self.end_ope_biases_batch[self.batch_idxes])
        true_idx = copy.deepcopy(self.ope_step_batch)
        true_idx[self.mask_job_finish_batch] -= 1
        t1 = convert_feat(self.feat_opes_batch[self.batch_idxes, 3, :], true_idx[self.batch_idxes])
        t2 = t0 - t1
        t2[self.mask_job_finish_batch[self.batch_idxes]] = 0
        self.feat_opes_batch[self.batch_idxes, 5, :] = convert_feat(t2, self.opes_appertain_batch[self.batch_idxes])
        # 更新能耗
        proc_powers = self.proc_powers_batch[self.batch_idxes, opes, mas]
        self.estimate_pp[self.batch_idxes, opes] = proc_powers
        self.feat_opes_batch[self.batch_idxes, 6, opes] = proc_times * proc_powers
        # padding工序特征处理
        mask1 = torch.all(self.proc_times_batch == 0, dim=-1)
        self.feat_opes_batch[mask1.unsqueeze(1).expand(-1, self.paras["ope_feat_dim"], -1)] = 0

        # 更新机器特征
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes],
                                                                          dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = end_act
        # Update partial schedule (gantt)
        mask2 = self.gantt_batch[self.batch_idxes, mas, :, 0] < 0
        indices = torch.argmax(mask2.int(), dim=1)
        self.gantt_batch[self.batch_idxes, mas, indices, 0] = start_act
        self.gantt_batch[self.batch_idxes, mas, indices, 1] = end_act
        self.gantt_batch[self.batch_idxes, mas, indices, 2] = opes.float()
        self.gantt_batch[self.batch_idxes, mas, indices, 3] = proc_powers
        mask3 = self.gantt_batch[self.batch_idxes, mas, :, 4] < 0
        indices1 = torch.argmax(mask3.int(), dim=1)
        differ = start_act - ma_rdy_time
        mask4 = 1 - torch.eq(ma_rdy_time, 0).int()
        differ *= mask4
        sb_start = torch.where(differ != 0, ma_rdy_time, -1)
        sb_end = torch.where(differ != 0, start_act, -1)
        self.gantt_batch[self.batch_idxes, mas, indices1, 4] = sb_start
        self.gantt_batch[self.batch_idxes, mas, indices1, 5] = sb_end
        self.feat_mas_batch[self.batch_idxes, 2, mas] += differ
        self.feat_mas_batch[self.batch_idxes, 3, mas] += differ * self.standby_powers_batch[self.batch_idxes, mas]
        # 更新全局特征
        self.feat_glo_batch[:, 0] = count_intersections(self.feat_opes_batch[:, 3, :], self.feat_opes_batch[:, 4, :],
                                                        self.performance_period)
        demand_pro = calculate_power_over_time(self.feat_opes_batch[:, 3, :],
                                               self.feat_opes_batch[:, 4, :],
                                               self.performance_period, self.estimate_pp)
        input_a = self.gantt_batch[..., 4]
        input_b = self.gantt_batch[..., 5]
        input_c = self.standby_powers_batch.unsqueeze(-1).expand_as(input_a)
        demand_sb = calculate_power_over_time(input_a.flatten(1), input_b.flatten(1), self.performance_period,
                                              input_c.flatten(1))
        demand_per = demand_pro + demand_sb
        self.feat_glo_batch[:, 1], self.feat_glo_batch[:, 2] = process_tensor(
            demand_per, self.hel - self.thr, self.hel)
        # feat_glo最小最大值归一化
        self.feat_glo_batch[:, 0] = self.feat_glo_batch[:, 0] / self.nums_opes
        self.feat_glo_batch[:, 1] = self.feat_glo_batch[:, 1] / len(self.performance_period)
        self.feat_glo_batch[:, 2] = self.feat_glo_batch[:, 2] / (self.hel - self.thr)
        self.feat_edge_batch = torch.stack((self.proc_times_batch, self.proc_powers_batch), dim=-1)

        # 更新机器队列前后关系
        pre_idx = torch.where(indices > 0, self.gantt_batch[self.batch_idxes, mas, indices - 1, 2],
                              self.num_opes - 1).int()
        relation_value = torch.where(indices > 0, True, False)
        self.solved_pre[self.batch_idxes, pre_idx, opes] = relation_value

        # Update other variable according to actions
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        # 更新适应度，计算奖励
        makespan = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        energy = torch.sum(self.feat_opes_batch[:, 6, :], dim=1) + torch.sum((input_b - input_a) * input_c, dim=(1, 2))
        deal_demand = torch.where(demand_per <= self.hel - self.thr, demand_per, self.hel)
        ir = torch.sum(deal_demand, dim=-1)
        self.f = torch.stack([makespan, energy, ir], dim=-1)
        weighted_sum = torch.sum(self.w * self.f, dim=-1)
        reward_batch = self.fitness - weighted_sum
        self.fitness = weighted_sum

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.feat_glo_batch,
                          self.feat_edge_batch, self.ope_ma_adj_batch, self.mask_job_finish_batch, self.ope_step_batch,
                          self.solved_pre)
        return self.state, reward_batch, self.done_batch

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.feat_glo_batch = copy.deepcopy(self.old_feat_glo_batch)
        self.feat_edge_batch = copy.deepcopy(self.old_feat_edge_batch)
        self.state = copy.deepcopy(self.old_state)
        self.proc_powers_batch = copy.deepcopy(self.old_proc_powers_batch)
        self.fitness = copy.deepcopy(self.old_fitness)

        self.batch_idxes = torch.arange(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)

        self.gantt_batch = -1 * torch.ones(size=(self.batch_size, self.num_mas, self.num_opes, 6))
        # 同一机器上的工序队列
        self.solved_pre = torch.full(size=(self.batch_size, self.num_opes, self.num_opes), dtype=torch.bool,
                                     fill_value=False)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        self.estimate_pt = copy.deepcopy(self.old_estimate_pt)
        self.estimate_pp = copy.deepcopy(self.old_estimate_pp)

        return self.state
