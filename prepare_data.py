def prepare_data(path):
    '''
    原始数据的处理
    将原始数据拆分成独立的案例
    '''
    with open(path) as f:
        data = []
        case = []
        for line in f.readlines():
            if '+' in line or 'instance' in line:
                # case不为空，一个案例收集完毕
                if case:
                    data.append(case)
                case = []
                continue
            else:
                case.append(line)

    # 拆分后的案例写入文件
    for i, case in enumerate(data):
        with open(f'data/case_{i}.txt', 'w') as f:
            for line in case:
                f.write(line)
    print(f'processed {len(data)} samples')

if __name__ == "__main__":
    prepare_data('./data/研究生-测试用例.txt')