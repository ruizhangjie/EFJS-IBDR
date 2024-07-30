import random


class CaseGenerator:
    '''
    FJSP instance generator
    '''

    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='../data/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save_1005 the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save_1005 path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = 5
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1
        self.proctime_per_ope_max = 6

    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas + 1), val))
        self.proc_time = []
        self.proc_power = []

        for i in range(len(self.nums_option)):
            # 控制时间（功率）差异性，同时更符合实际情况
            proc_time_ope = sorted(random.sample(range(self.proctime_per_ope_min, self.proctime_per_ope_max + 1),
                                                 self.nums_option[i]))
            proc_power_ope = [round(36 / (x * x), 3) for x in proc_time_ope]
            indexes = list(range(self.nums_option[i]))
            random.shuffle(indexes)
            self.proc_time = self.proc_time + [proc_time_ope[i] for i in indexes]
            # 生成各工序在对应可选机器上的功率序列
            self.proc_power = self.proc_power + [proc_power_ope[i] for i in indexes]
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        lines_power = []
        lines_power_doc = []  # 功率部分作为文件生成
        lines_power.append('##==##\n')
        lines_power_doc.append('##==##')
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            line_power = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i] + self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    line_power.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i] + idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i] + idx_ope])
                    line_power.append(self.nums_option[self.num_ope_biass[i] + idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    line_power.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    line_power.append(self.proc_power[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    str_line_power = " ".join([str(val_power) for val_power in line_power])
                    lines_power.append(str_line_power + '\n')
                    lines_power_doc.append(str_line_power)
                    break
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas,
                                                              str.zfill(str(idx + 1), 3)), 'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            for i in range(len(lines_power_doc)):
                print(lines_power_doc[i], file=doc)
            doc.close()
        return lines, lines_power


if __name__ == '__main__':
    from fjsp_env import FJSPEnv
    import json
    import torch
    import os
    from schedule_model import Memory, HGNNScheduler
    import numpy as np

    case_generator = CaseGenerator(20, 10, 10, 14,
                                   flag_same_opes=False, path='data_dev/2010/', flag_doc=True)
    for i in range(10):
        case_generator.get_case(i)
