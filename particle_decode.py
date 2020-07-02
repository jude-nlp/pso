#-*- coding: UTF-8 -*- 
import numpy as np
import math

def get_min_time(schedulable, time1, time2):
    # 在所有未安排的操作中，获取最小的最早开始时间和最小的最早结束时间
    tmp1 = []
    tmp2 = []
    for i,j in schedulable:
        tmp1.append(time1[i - 1][j - 1])
        tmp2.append(time2[i - 1][j - 1])
    return [min(tmp1), min(tmp2)]

def get_operation_priority(n, m, particle):
    '''
    根据粒子的位置信息，获取操作的优先级
    sche: n个工件的工序矩阵
    times: sche中工序对应的时间
    particle: 粒子
    本方法参照论文 'Particle swarm optimization algorithm applied to scheduling problems' Algorithm 2
    '''

    X = []
    pai = []
    for i, pos in enumerate(particle, start=1):
        X.append([i, pos])
    X.sort(key=lambda x:x[1],reverse=False)

    for i in range(n):
        tmp = [[i + 1]] * m
        pai.extend(tmp)
    X_pai = np.concatenate((X, pai), axis=1)
    X_pai = X_pai[np.argsort(X_pai[:,0])]

    order = list(X_pai[:,2].astype(np.uint8))
    count = np.zeros(n + 1)
    op = []
    for k in order:
        count[k] += 1      
        op.append([k, count[k].astype(np.int8)]) 
    
    return op

def decode(n, m, sche, time, particle, delta):
    '''
    m * n次迭代，安排所有的m * n道工序
    scheduled: 已安排好的 operation
    schedulable: 待安排的 operation, 优先级随索引递增而递减
    delta: delta=0 生成non-delay schedules; delta→1 生成active schedules
    '''
    # 最早开始/结束时间 初始化
    earliest_start_time = np.zeros((n, m))
    earliest_end_time = np.zeros((n, m)) 
    for i in range(n):
        for j in range(m):
            earliest_start_time[i][j] = sum(time[i][:j])
    for i in range(n):
        for j in range(m):
            earliest_end_time[i][j] = sum(time[i][:j+1])

    scheduled = []
    schedulable = get_operation_priority(n, m, particle)
    
    for _ in range(m*n): 
        # 最早开始/结束时间的最小值
        start_min, end_min = get_min_time(schedulable, earliest_start_time, earliest_end_time)
        machine_id = -1
        end_time = -1
        # 按照操作优先级，依次寻找符合条件的点
        for j in range(len(schedulable)):
            k = schedulable[j][0] - 1
            l = schedulable[j][1] - 1
            if earliest_start_time[k][l] <= start_min + delta * (end_min - start_min):
                scheduled.append([k + 1, l + 1])
                schedulable.remove([k + 1, l + 1])
                machine_id = sche[k][l]
                end_time = earliest_end_time[k][l]
                break
            else:
                continue

        # 更新 earliest_start_time 和 earliest_end_time
        assert (machine_id >= 0), 'machine_id 错误'
        assert (end_time >= 0), 'end_time 错误'
        for o in schedulable:
            p = o[0] - 1
            q = o[1] - 1
            # 如果剩余安排操作要用当前分配的机器，并且最早开始时间要早于该机器释放的时间
            if sche[p][q] == machine_id and (earliest_start_time[p][q] < end_time):
                delta_time = end_time - earliest_start_time[p][q]
                earliest_start_time[p][q:] += delta_time
                earliest_end_time[p][q:] += delta_time

    total_time = np.max(earliest_end_time)

    return total_time, scheduled, earliest_start_time, earliest_end_time