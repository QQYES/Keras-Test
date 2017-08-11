def add(line):
    if isinstance(line, str):
        num_list = line.split(' ')
        return int(num_list[0]) + int(num_list[1])


print(add('1 2'))
