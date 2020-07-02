#-*- coding: UTF-8 -*- 
import numpy as np
from particle_decode import decode
from tqdm import tqdm
from math import inf
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Particle:
    def __init__(self, params, n, m, sche, times):
        self.params = params
        self.n = n                  # 工件数
        self.m = m                  # 机器数
        self.sche = sche            # 工序矩阵
        self.times = times          # 工序对应的时间矩阵
        self.dimension = n * m      # 维度信息
        self.position = np.zeros(self.dimension)    # 粒子位置
        self.velocity = np.zeros(self.dimension)    # 粒子速度，初始化为 0
        self.p_best = inf          # 粒子历史最佳位置对应的加工时间
        self.p_best_position = []   # 粒子最佳位置

        self.pos_init()

    # 粒子位置的初始化
    def pos_init(self):
        for i in range(self.dimension):
            r = np.random.rand(1)
            self.position[i] = self.params.pos_min + (self.params.pos_max - self.params.pos_min) * r

    # 计算当前粒子的适应度
    def evaluate(self):
        return decode(self.n, self.m, self.sche, self.times, self.position, self.params.delta)

    # 更新速度
    def update_velocity(self, g_best_position):
        w = self.params.inertia_weight
        c1 = self.params.cognative_c1
        c2 = self.params.social_c2

        for i in range(self.dimension):
            r1 = np.random.rand()
            r2 = np.random.rand()

            cognitive = c1 * r1 * (self.p_best_position[i] - self.position[i])
            social = c2 * r2 * (g_best_position[i] - self.position[i])
            new_veiocity = float(w * self.velocity[i] + cognitive + social)

            # 限制速度值介于[vel_min, vel_max]之间
            if new_veiocity < self.params.vel_min:
                self.velocity[i] = self.params.vel_min
            elif new_veiocity > self.params.vel_max:
                self.velocity[i] = self.params.vel_max
            else:
                self.velocity[i] = new_veiocity

    # 更新位置
    def update_position(self):
        for i in range(self.dimension):
            new_pos = self.position[i] + self.velocity[i]

            # 限制位置介于[pos_min, pos_max]之间
            if new_pos < self.params.pos_min:
                self.position[i] = float(self.params.pos_min)
            elif new_pos > self.params.pos_max:
                self.position[i] = float(self.params.pos_max)
            else:
                self.position[i] = float(new_pos)

class PSO:
    def __init__(self, params, n, m, sche, times):
        self.params = params
        # 案例信息参数
        self.n = n
        self.m = m
        self.dimension = n * m
        self.sche = sche
        self.times = times
        self.g_best = inf
        self.g_best_position = np.zeros(self.dimension)

        # 定义种群并初始化
        self.swarm = []
        for _ in range(self.params.particle_num):
            self.swarm.append(Particle(self.params, n, m, sche, times))

    # 返回随机的颜色
    def random_color(self):
        colArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for _ in range(6):
            color += colArr[random.randint(0,14)]
        return "#"+color
    
    # 根据最终的g_best绘制”甘特图“
    def draw(self, scheduled, start, end, path):
        color = []
        for _ in range(self.n):
            color.append(self.random_color())

        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['savefig.dpi'] = 900
        plt.title("Gantt Chart", fontsize=18)
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Machine ID", fontsize=18)
        plt.yticks(range(0, self.m+1, 1))
        for p, q in scheduled:
            plt.barh(y=self.sche[p-1][q-1], width=(end[p-1][q-1] - start[p-1][q-1]), height=0.5, left=start[p-1][q-1], color=color[p-1])
            plt.text(start[p-1][q-1], self.sche[p-1][q-1], 'J%d,%d' % (p, q), fontdict={'color':  'white'}, size=50/self.n)

        path = path + '_Gantt.png'
        plt.savefig(path)
        print('Saved global best Gantt chart  to %s' % path)

    # 保存最佳解决方案
    def save_result(self, path, scheduled, total_time):
        path = path + '_solution.txt'
        with open(path, 'w') as f:
            f.write('最短用时:%d' % total_time + '\n')
            for item in scheduled:
                f.write('工件ID:%d, 工序:%d' % (item[0], item[1]) + '\n')
        print('Saved global best solution  to %s' % path)

    def train(self):
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)
        print('Start training...')
        pbar = tqdm(range(self.params.epoch_size))
        for epoch in pbar:
            # 遍历种群，并计算粒子适应度(fitness)
            for i in range(self.params.particle_num):
                total_time, _, _, _ = self.swarm[i].evaluate()

                # 更新 p_best
                if total_time < self.swarm[i].p_best:
                    self.swarm[i].p_best = total_time
                    self.swarm[i].p_best_position = list(self.swarm[i].position)
                else:
                    continue
                # 更新 g_best
                if total_time < self.g_best:
                    self.g_best = total_time
                    self.g_best_position = list(self.swarm[i].position)
            # 遍历种群，更新粒子速度/位置
            for i in range(self.params.particle_num):
                self.swarm[i].update_velocity(self.g_best_position)
                self.swarm[i].update_position()

            pbar.set_description('epoch %d: min global total_time %.4f' % (epoch, self.g_best))
        
        print('End training...')
        total_time, scheduled, start_time, end_time = decode(self.n, self.m, self.sche, self.times, self.g_best_position, self.params.delta)
        
        # 保存最佳解决方案以及对应“甘特图”
        path = 'output/case_%d' % self.params.case_id
        self.save_result(path, scheduled, total_time)
        self.draw(scheduled, start_time, end_time, path)

        print('Best result: %.2f' % self.g_best)
