import pandas as pd
import ssq_data_import_02 as ssq2

def new_balls():
    # 调用DoubleColorBalls 函数
    ssq2.DoubleColorBalls()

    # 读取原数据及新数据
    old_ball = pd.read_csv("balls_data.csv")
    new_ball = pd.read_csv("balls_data1.csv")

    # 比较 new_ball 和 old_ball 中的 "num" 列，找出 new_ball 中比 old_ball 中新的数据行
    new_rows = new_ball[new_ball['num'] > old_ball['num'].max()]

    # 将新的数据行添加到 old_ball 的底部，并按 "num" 列排序
    old_ball = pd.concat([old_ball, new_rows]).sort_values(by='num').reset_index(drop=True)

    old_ball.to_csv("balls_data1.csv", index=False)

if __name__ == '__main__':
    new_balls()