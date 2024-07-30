import copy
import random

import numpy as np


class Refer:
    def __init__(self, matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, num_ope_biases, nums_ope, slot,
                 matrix_sb_power, thr, p_T=0.4, p_P=0.3, p_N=0.2, p_R=0.2, p_L=0.2):
        self.nums_ope = nums_ope.clone().cpu().numpy()
        self.matrix_proc_time = matrix_proc_time.clone().cpu().numpy()
        self.matrix_proc_power = matrix_proc_power.clone().cpu().numpy()
        self.matrix_ope_ma_adj = matrix_ope_ma_adj.clone().cpu().numpy()
        self.matrix_sb_power = matrix_sb_power.clone().cpu().numpy()
        self.num_ope_biases = num_ope_biases.clone().cpu().numpy()
        self.slot = slot.clone().cpu().numpy()
        self.thr = thr
        self.all_opes = np.sum(self.nums_ope)
        self.num_mas = self.matrix_proc_time.shape[-1]
        self.pop_size = 30
        self.p_T, self.p_N, self.p_R, self.p_L, self.p_P = p_T, p_N, p_R, p_L, p_P
        dim0 = self.matrix_sb_power.shape[1]
        if dim0 > self.all_opes:
            self.matrix_sb_power = self.matrix_sb_power[:, :self.all_opes]


    def encode_ROS(self):
        b = [0] * self.all_opes
        start = 0
        for i, num in enumerate(self.nums_ope):
            b[start:start + num] = [i] * num
            start += num
        random.shuffle(b)
        return b

    def encode_TOS(self):
        # 平均加工时间最小
        result = []
        ope_step = copy.deepcopy(self.num_ope_biases)
        b = np.sum(self.matrix_proc_time, axis=1) / np.sum(self.matrix_proc_time != 0, axis=1).astype(float)
        for i in range(self.all_opes):
            c = b[ope_step]
            # 找到最小元素的索引
            min_indices = np.where(c == np.min(c))[0]
            random_min_index = np.random.choice(min_indices)
            result.append(random_min_index)
            b[ope_step[random_min_index]] = float('inf')
            if random_min_index != len(ope_step) - 1:
                if ope_step[random_min_index] != self.num_ope_biases[random_min_index + 1] - 1:
                    ope_step[random_min_index] += 1
            else:
                if ope_step[random_min_index] != self.all_opes - 1:
                    ope_step[random_min_index] += 1
        return result

    def encode_POS(self):
        # 平均加工功率最小
        result = []
        ope_step = copy.deepcopy(self.num_ope_biases)
        b = np.sum(self.matrix_proc_power, axis=1) / np.sum(self.matrix_proc_power != 0, axis=1).astype(float)
        for i in range(self.all_opes):
            c = b[ope_step]
            # 找到最小元素的索引
            min_indices = np.where(c == np.min(c))[0]
            random_min_index = np.random.choice(min_indices)
            result.append(random_min_index)
            b[ope_step[random_min_index]] = float('inf')
            if random_min_index != len(ope_step) - 1:
                if ope_step[random_min_index] != self.num_ope_biases[random_min_index + 1] - 1:
                    ope_step[random_min_index] += 1
            else:
                if ope_step[random_min_index] != self.all_opes - 1:
                    ope_step[random_min_index] += 1
        return result

    def encode_NOS(self):
        # 剩余工序最多
        result = []
        num_step = copy.deepcopy(self.nums_ope)
        for i in range(self.all_opes):
            max_value = np.max(num_step)
            max_indices = np.where(num_step == max_value)[0]
            random_min_index = np.random.choice(max_indices)
            result.append(random_min_index)
            num_step[random_min_index] -= 1
        return result

    def encode_RMS(self):
        result = []
        for i in range(self.all_opes):
            non_zero_indices = np.where(self.matrix_proc_time[i] > 0)[0]
            if len(non_zero_indices) > 0:
                result.append(np.random.choice(non_zero_indices))
        return result

    def encode_TMS(self):
        # 加工时间最小
        result = []
        for i in range(self.all_opes):
            arr = self.matrix_proc_time[i]
            non_zero_elements = arr[arr != 0]
            min_value = np.min(non_zero_elements)
            min_indices = np.where(arr == min_value)[0]
            result.append(np.random.choice(min_indices))
        return result

    def encode_PMS(self):
        # 加工功率最小
        result = []
        for i in range(self.all_opes):
            arr = self.matrix_proc_power[i]
            non_zero_elements = arr[arr != 0]
            min_value = np.min(non_zero_elements)
            min_indices = np.where(arr == min_value)[0]
            result.append(np.random.choice(min_indices))
        return result

    def encode_LMS(self):
        # 全局选择策略
        result = [0] * self.all_opes
        cum_load = np.zeros(self.num_mas, dtype=np.int64)
        job_list = [_ for _ in range(self.nums_ope.shape[0])]
        proc_time = copy.deepcopy(self.matrix_proc_time)
        proc_time[proc_time == 0] = float('inf')
        all_job = len(job_list)
        while job_list:
            i = random.choice(job_list)  # 随机选择一个工件
            start_op = self.num_ope_biases[i]
            if i != all_job - 1:
                end_op = self.num_ope_biases[i + 1]
            else:
                end_op = self.all_opes
            a = proc_time[start_op:end_op]
            for j in range(a.shape[0]):
                b = a[j] + cum_load
                min_index_candidates = np.where(b == np.min(b))[0]
                random_min_index = np.random.choice(min_index_candidates)
                result[start_op + j] = random_min_index
                cum_load[random_min_index] += a[j, random_min_index]
            job_list.remove(i)  # 将选择后的工件从未选工件集中移除
        return result

    def Pop_Gene(self, size):
        Pop = []
        for i in range(size):
            methods = [self.encode_TOS, self.encode_POS, self.encode_NOS, self.encode_ROS]
            probabilities = [self.p_T, self.p_P, self.p_N, self.p_R]
            chosen_method = random.choices(methods, probabilities)[0]
            os_list = chosen_method()
            methods1 = [self.encode_TMS, self.encode_PMS, self.encode_LMS, self.encode_RMS]
            probabilities1 = [self.p_T, self.p_P, self.p_L, self.p_R]
            chosen_method1 = random.choices(methods1, probabilities1)[0]
            ms_list = chosen_method1()
            item = os_list + ms_list
            Pop.append(item)
        return np.array(Pop)

    def decode_Ma(self, chromosome):
        matrix = np.zeros((self.all_opes, 3), dtype=np.float32)
        start = -99 * np.ones((self.num_mas, self.all_opes), dtype=np.float32)
        sb_start = -99 * np.ones((self.num_mas, self.all_opes), dtype=np.float32)
        end = np.zeros_like(start, dtype=np.float32)
        sb_end = np.zeros_like(start, dtype=np.float32)
        demand = np.zeros_like(start, dtype=np.float32)
        op_idx = -99 * np.ones_like(start, dtype=np.int64)
        Os = chromosome[:self.all_opes]
        Ms = chromosome[self.all_opes:]
        # 先解码MS部分
        for i in range(self.all_opes):
            ma = Ms[i]
            time = self.matrix_proc_time[i, ma]
            power = self.matrix_proc_power[i, ma]
            matrix[i, 0] = ma
            matrix[i, 1] = time
            matrix[i, 2] = power
        step_ope_biases = copy.deepcopy(self.num_ope_biases)
        for i in range(self.all_opes):
            x = self.cal_demand(demand, start, end) + self.cal_demand(self.matrix_sb_power, sb_start, sb_end)
            y = (np.sum(x, axis=0))[self.slot.astype(int)]
            z0 = np.sum(y > self.thr)  # 解码前超过阈值的时间窗数量
            job = Os[i]
            op = step_ope_biases[job]
            ma = int(matrix[op, 0])
            process_time = matrix[op, 1]
            process_power = matrix[op, 2]
            start_a = start[ma]
            end_a = end[ma]
            sb_start_a = sb_start[ma]
            sb_end_a = sb_end[ma]
            op_idx_a = op_idx[ma]
            power_a = demand[ma]
            job_release = 0
            if op not in self.num_ope_biases:
                job_release = end[op_idx == op - 1]
            ma_release = np.max(end_a)
            possiblePos = np.where(job_release < start_a)[0]
            flag = False
            if len(possiblePos) == 0:
                st_time = self.putInTheEnd(op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a, op_idx_a,
                                           power_a, process_time, process_power)
            else:
                idxLegalPos, legalPos, startTimeEarlst = self.calLegalPos(possiblePos, start_a, end_a, job_release,
                                                                          process_time)
                if len(legalPos) == 0:
                    st_time = self.putInTheEnd(op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a,
                                               op_idx_a, power_a, process_time, process_power)
                else:
                    st_time, earlstPos = self.putInBetween(op, idxLegalPos, legalPos, startTimeEarlst, start_a, end_a,
                                                           sb_start_a, sb_end_a, op_idx_a, power_a, process_time,
                                                           process_power)
                    flag = True

            # 计算各时间窗能耗
            x1 = self.cal_demand(demand, start, end) + self.cal_demand(self.matrix_sb_power, sb_start, sb_end)
            y1 = (np.sum(x1, axis=0))[self.slot.astype(int)]
            z1 = np.sum(y1 > self.thr)
            if z1 > z0:
                if np.isin(st_time, self.slot):
                    # 超过阈值时间窗增加且开始时间位于性能期内
                    if flag:
                        # 插入到最后
                        # 更新时间、索引和功率等信息
                        a = np.where(start_a == -99)[0][0] - 1
                        b = end_a[a]
                        self.rearrange_elements(start_a, earlstPos, -99)
                        self.rearrange_elements(end_a, earlstPos, 0)
                        if b >= self.slot[-1] + 1:
                            start_a[a] = b
                            end_a[a] = b + process_time
                        else:
                            start_a[a] = self.slot[-1] + 1
                            end_a[a] = self.slot[-1] + 1 + process_time
                            pos = np.where(sb_start_a == -99)[0][0]
                            sb_start_a[pos] = b
                            sb_end_a[pos] = self.slot[-1] + 1

                        self.rearrange_elements(op_idx_a, earlstPos, -99)
                        self.rearrange_elements(power_a, earlstPos, 0)
                        et = st_time + process_time
                        # 更新空闲时间
                        if earlstPos == 0:
                            st = start_a[0]
                            if et != st:
                                pos = np.where(sb_start_a == et)[0][0]
                                sb_start_a[pos] = -99
                                sb_end_a[pos] = 0
                        else:
                            pre_e = end_a[earlstPos - 1]
                            sub_s = start_a[earlstPos]
                            if pre_e == st_time:
                                if et == sub_s:
                                    pos = np.where(sb_start_a == -99)[0][0]
                                    sb_start_a[pos] = pre_e
                                    sb_end_a[pos] = sub_s
                                else:
                                    pos = np.where(sb_start_a == et)[0][0]
                                    sb_start_a[pos] = pre_e
                            else:
                                pos = np.where(sb_end_a == st_time)[0][0]
                                sb_end_a[pos] = sub_s
                                if et != sub_s:
                                    pos1 = np.where(sb_start_a == et)[0][0]
                                    sb_start_a[pos1] = -99
                                    sb_end_a[pos1] = 0

                    else:
                        # 移动到性能期外
                        pos = np.where(start_a == -99)[0][0] - 1
                        st = self.slot[-1] + 1
                        et = st + process_time
                        start_a[pos] = st
                        end_a[pos] = et
                        if pos > 0:
                            pre_e = end_a[pos - 1]
                            if pre_e == st_time:
                                pos1 = np.where(sb_start_a == -99)[0][0]
                                sb_start_a[pos1] = pre_e
                                sb_end_a[pos1] = st
                            else:
                                pos1 = np.where(sb_start_a == pre_e)[0][0]
                                sb_end_a[pos1] = st

            step_ope_biases[job] += 1

        x = self.cal_demand(demand, start, end) + self.cal_demand(self.matrix_sb_power, sb_start, sb_end)
        s = copy.deepcopy(sb_start)
        s[s == -99] = 0  # 方便计算TPC
        Cmax = np.max(end)
        TPC = np.sum((end - start) * demand) + np.sum((sb_end - s) * self.matrix_sb_power)
        x1 = (np.sum(x, axis=0))[self.slot.astype(int)]
        x2 = x1[x1 <= self.thr]
        IR = np.sum(x2)
        return Cmax.item(), TPC.item(), IR.item()

    def putInTheEnd(self, op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a, op_idx_a, power_a,
                    process_time, process_power):
        index = np.where(start_a == -99)[0][0]
        startTime_a = max(job_release, ma_release)
        start_a[index] = startTime_a
        op_idx_a[index] = op
        power_a[index] = process_power
        end_a[index] = startTime_a + process_time
        if index > 0:
            st = end_a[index - 1]
            if st < startTime_a:
                idx1 = np.where(sb_start_a == -99)[0][0]
                sb_start_a[idx1] = st
                sb_end_a[idx1] = startTime_a
        return startTime_a

    def calLegalPos(self, possiblePos, start_a, end_a, job_release, process_time):
        part_start = end_a[possiblePos[:-1]]
        if possiblePos[0] != 0:
            t1 = end_a[possiblePos[0] - 1]
            t2 = max(job_release, t1)
            startTimeEarlst = np.insert(part_start, 0, t2)
        else:
            t = max(job_release, 0)
            startTimeEarlst = np.insert(part_start, 0, t)
        dur = start_a[possiblePos] - startTimeEarlst
        idxLegalPos = np.where(dur >= process_time)[0]  # possiblePos中的下标
        legalPos = np.take(possiblePos, idxLegalPos)  # start_a中的下标
        return idxLegalPos, legalPos, startTimeEarlst

    def putInBetween(self, op, idxLegalPos, legalPos, startTimeEarlst, start_a, end_a, sb_start_a, sb_end_a, op_idx_a,
                     power_a, process_time, process_power):
        earlstIdx = idxLegalPos[0]
        earlstPos = legalPos[0]
        startTime_a = startTimeEarlst[earlstIdx]
        start_a[:] = np.insert(start_a, earlstPos, startTime_a)[:-1]
        end_a[:] = np.insert(end_a, earlstPos, startTime_a + process_time)[:-1]
        op_idx_a[:] = np.insert(op_idx_a, earlstPos, op)[:-1]
        power_a[:] = np.insert(power_a, earlstPos, process_power)[:-1]
        st = startTime_a + process_time
        if earlstPos == 0:
            et = start_a[1]
            if st != et:
                pos = np.where(sb_start_a == -99)[0][0]
                sb_start_a[pos] = st
                sb_end_a[pos] = et
        else:
            pre_e = end_a[earlstPos - 1]
            sub_s = start_a[earlstPos + 1]
            pos = np.where(sb_start_a == pre_e)[0][0]
            if pre_e == startTime_a:
                if sub_s != st:
                    sb_start_a[pos] = st
                else:
                    sb_start_a[pos] = -99
                    sb_end_a[pos] = 0
            else:
                sb_end_a[pos] = startTime_a
                if sub_s != st:
                    pos1 = np.where(sb_start_a == -99)[0][0]
                    sb_start_a[pos1] = st
                    sb_end_a[pos1] = sub_s
        return startTime_a, earlstPos

    def cal_demand(self, power, start, end):
        dim0 = np.shape(start)[0]
        dim1 = np.shape(start)[1]
        dim2 = np.shape(power)[1]
        a = np.tile(start, (96, 1, 1))
        b = np.tile(end, (96, 1, 1))
        c = np.tile(power, (96, 1, 1))
        steps = np.arange(96, dtype=np.float32)
        temp_array = np.tile(steps, (dim0, 1))
        d = np.tile(temp_array, (dim1, 1, 1)).transpose(2, 1, 0)
        e = np.where(a <= d, d, -1)
        e = np.where(e < b, e, -1)
        f = np.where(e != -1, 1, 0)
        g = f * c
        result = np.sum(g, axis=-1).T
        return result

    def rearrange_elements(self, arr, selected_index, flag):

        # Find the boundary where -99 starts
        boundary = np.where(arr == flag)[0][0]

        # Get the value at the selected index
        selected_value = arr[selected_index]

        # Move the elements before the selected index
        arr[selected_index:boundary - 1] = arr[selected_index + 1:boundary]

        # Place the selected element at the boundary - 1
        arr[boundary - 1] = selected_value

        return arr

    def get_z(self):
        a, b, c = float('inf'), float('inf'), float('inf')
        chs_list = self.Pop_Gene(self.pop_size)
        for i in range(self.pop_size):
            chromosome = chs_list[i]
            Cmax, TPC, IR = self.decode_Ma(chromosome)
            if a > Cmax:
                a = Cmax
            if b > TPC:
                b = TPC
            if c > IR:
                c = IR
        return a, b, c
