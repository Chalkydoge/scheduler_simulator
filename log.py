import csv

required_fields = [
    'EXEC', 'IPC', 'FREQ', 'AFREQ', 'L3MISS', 'L2MISS', 'L3HIT', 'L2HIT',
    'L3MPI', 'L2MPI', 'L3OCC', 'READ', 'WRITE', 'LLCRDMISSLAT', 'INST'
]
     
def parse_skyt_line(line):
    fields = line.strip().split()
    if len(fields) < 16 or fields[0] != 'SKT':
        return None
    try:
        data = {
            'EXEC': fields[2],
            'IPC': fields[3],
            'FREQ': fields[4],
            'AFREQ': fields[5],
            'L3MISS': int(fields[6]) * 1000 if fields[7] == 'K' else int(fields[6]),
            'L2MISS': int(fields[8]) * 1000 if fields[9] == 'K' else int(fields[8]),
            'L3HIT': fields[10],
            'L2HIT': fields[11],
            'L3MPI': fields[12],
            'L2MPI': fields[13],
            'L3OCC': fields[14],
        }
        return data
    except IndexError:
        return None

def parse_instructions_retired(line):
    parts = line.split(';')
    for part in parts:
        if 'Instructions retired:' in part:
            inst_part = part.split(':')[1].strip()
            inst_value = ' '.join(inst_part.split()[:2])
            
            unit = inst_value[-1]
            if unit == 'K':
                inst_value = int(float(inst_value[:-1]) * 1000)
            elif unit == 'M':
                inst_value = int(float(inst_value[:-1]) * 1000000)
            elif unit == 'G':
                inst_value = int(float(inst_value[:-1]) * 1000000000)
            elif unit == 'T':
                inst_value = int(float(inst_value[:-1]) * 1000000000000)
            else:
                inst_value = int(inst_value)
            return inst_value
    return None

def parse_mem_line(line):
    fields = line.strip().split()
    if len(fields) < 9 or fields[0] != 'SKT':
        return None
    return {
        'READ': fields[2],
        'WRITE': fields[3],
        'LLCRDMISSLAT': fields[8]
    }


def main():
    current_data = {}
    in_mem_section = False

    with open('results.log', 'r') as f, open('node_output.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=required_fields)
        writer.writeheader()

        for line in f:
            line = line.strip()
            if in_mem_section:
                if line.startswith('SKT'):
                    mem_data = parse_mem_line(line)
                    if mem_data:
                        current_data.update(mem_data)
                        in_mem_section = False
                        # 检查是否所有字段都存在
                        if all(key in current_data for key in required_fields):
                            writer.writerow(current_data)
                            current_data = {}
                elif line.startswith('---'):
                    continue
            else:
                if line.startswith('SKT'):
                    skt_data = parse_skyt_line(line)
                    if skt_data:
                        current_data.update(skt_data)
                elif 'Instructions retired:' in line:
                    inst = parse_instructions_retired(line)
                    if inst:
                        current_data['INST'] = inst
                elif line.startswith('MEM (GB)->'):
                    in_mem_section = True

        # 处理文件末尾可能未写入的数据
        if all(key in current_data for key in required_fields):
            writer.writerow(current_data)


def get_avg():
    # 读取 CSV 文件中的数据
    data = []
    with open('node_output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取表头
        exec_index = headers.index('EXEC')
        for row in reader:
            # 过滤掉 EXEC 列中大于 1 的行
            if float(row[exec_index]) <= 1:
                data.append([float(value) for value in row])


    # 计算每列的平均值
    averages = []
    num_rows = len(data)
    num_cols = len(data[0])

    for col in range(num_cols):
        col_sum = sum(row[col] for row in data)
        col_avg = col_sum / num_rows
        averages.append(col_avg)

    # 将平均值添加到 CSV 文件的最后一行
    with open('node_output_avg.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(averages)

if __name__ == "__main__":
    main()
    get_avg()