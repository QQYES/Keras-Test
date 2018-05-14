file = open("onduty.csv", encoding='utf-8')
while 1:
    line = file.readline()
    if not line:
        break
    fields = line.split(",")
    if (len(fields) < 4):
        pass
    duty_name = fields[0]
    duty_type = fields[1]
    duty_date = fields[2]
    print('line:{}'.format(line))
    print(duty_name, duty_type, duty_date)
    print('duty_name:{},duty_type:{},duty_date:{}'.format(duty_name, duty_type, duty_date))
