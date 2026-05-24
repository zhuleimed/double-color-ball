import csv
import ssq_data_import_01 as ssq

def DoubleColorBalls():
    # 创建双色球对象
    ball = ssq.DoubleColorBall()
    # 获取双色球数据
    ball.getBall()

    # 定义输入和输出文件名
    input_file = 'balls_data1.txt'
    output_file = 'balls_data1.csv'

    # 创建一个列表用于存储每行的数据
    data = []

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行尾的换行符，并按空格分割成列
            row = line.strip().split()
            # 将行数据添加到列表中
            data.append(row)

    # 对数据进行排序，先按年份升序，再按期数升序
    data.sort(key=lambda x: (int(x[0]) // 1000, int(x[0]) % 1000))

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建CSV写入对象
        csvwriter = csv.writer(csvfile)
        # 写入列名
        csvwriter.writerow(['num', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue1'])
        # 写入数据
        csvwriter.writerows(data)


if __name__ == '__main__':
    DoubleColorBalls()