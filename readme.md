# Particle swarm optimization algorithm applied to scheduling problems

**run**

```shell
python main.py
--case_id	0		# 案例id
--epoch_size 50			# 迭代数
--delta 0.4			# 控制机器空闲/忙碌程度
--particle_num 24		# 粒子数目
--pos_max	100	        # 粒子位置最大值
--pos_min 0			# 粒子位置最小值
--vel_max	1		# 粒子速度最大值
--vel_min -1		        # 粒子速度最小值
--inertia_weight 1	        # 惯性权重
--cognative_c1 2		# 感知参数
--social_c2 2		        # 社会参数
```

