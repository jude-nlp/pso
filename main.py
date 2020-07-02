#-*- coding: UTF-8 -*- 
import numpy as np
import argparse
import os
from pso import PSO

def get_parser():
    '''
    生成参数 parser
    '''

    # parse parameters
    parser = argparse.ArgumentParser(description="PSO")
    # 主参数
    parser.add_argument("--case_id", type=int, default="-1",
                    help="test case id")
    parser.add_argument("--epoch_size", type=int, default="50",
                help="epoch size")
    parser.add_argument("--delta", type=float, default="0",
                help="the bound on the length of time a machine is allowed to remain idle")
    parser.add_argument("--seed", type=int, default="100",
                help="random seed")
    # 粒子群相关参数
    parser.add_argument("--particle_num", type=int, default="24",
            help="particle num")
    # 位置、速度限制区间
    parser.add_argument("--pos_max", type=float, default="100",
            help="max position of a particle")
    parser.add_argument("--pos_min", type=float, default="0",
            help="min position of a particle")
    parser.add_argument("--vel_max", type=float, default="1",
            help="max velocity of a particle")
    parser.add_argument("--vel_min", type=float, default="-1",
            help="min velocity of a particle")
    # 惯性权重w 感知参数c1  社会参数c2
    parser.add_argument("--inertia_weight", type=float, default="1",
            help="how much to weigh the previous velocity")
    parser.add_argument("--cognative_c1", type=float, default="2",
            help="cognative constant")
    parser.add_argument("--social_c2", type=float, default="2",
            help="social constant")

    return parser

def load_data(path):
    '''
    返回：工件数n、 机器数m、 工件加工顺序矩阵sche、 工件加工时间矩阵times
    '''
    print('loading data from %s' % path)
    lines = []
    with open(path) as f:
        for line in f.readlines():
            lines.append(line.strip())
    # 加载工件数n和机器数m
    n, m = list(map(int, lines[0].split()))
    # 加载工序矩阵，对应时间矩阵
    sche = []
    times = []
    for i in range(n):
        machine = []
        t = []
        idx = i + 1
        line = lines[idx].strip().split()
        for j in range(len(line)):
            if (j % 2 == 0):
                machine.append(int(line[j]))
            else:
                t.append(int(line[j]))
        sche.append(machine)
        times.append(t)
    return n, m, sche, times

def main():
    # 获取 parser
    parser = get_parser()
    params = parser.parse_args()
    print(params)

    # 加载数据
    assert (params.case_id >= 0), '未指定案例ID --case_id'
    path = 'data/case_%d.txt' % params.case_id
    n, m, sche, times = load_data(path)

    # 模型加载，训练
    model = PSO(params, n, m, sche, times)
    model.train()

if __name__ == "__main__":
    main()