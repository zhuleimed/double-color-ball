# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import optuna
import new_balls_cycle_obtain_save as balls_save

# 设置Pandas显示选项
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# 调用balls_save.new_balls()函数，生成新的数据集
balls_save.new_balls()

# 读取数据
df1 = pd.read_csv('balls_data.csv')  # 读取CSV文件
df1 = df1.drop(columns=['num'])  # 删除不需要的列
df1 = df1.dropna()  # 删除缺失值
df1 = df1.reset_index(drop=True)  # 重置索引
df1.columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'nb']  # 重命名列
df = df1
print(df.shape)  # 打印DataFrame的形状

# 提取特征函数
def extract_features(df):
    features = pd.DataFrame()
    features[('mean')] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].mean(axis=1)  # 计算每一行的特征：均值、标准差、最大值、最小值、范围、求和。
    features[('std')] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].std(axis=1)
    features[('max')] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].max(axis=1)
    features[('min')] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].min(axis=1)
    features['range'] = features['max'] - features['min']
    features['sum'] = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].sum(axis=1)
    return features

# 构建训练数据集
X = extract_features(df)  # 提取特征
y = df['nb']  # 提取目标变量

# 对目标变量进行Label Encoding
label_encoder = LabelEncoder()  # 创建LabelEncoder对象
y_encoded = label_encoder.fit_transform(y)  # 对目标变量进行Label Encoding（分类）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  # 划分训练集和测试集，其中X_train是训练集的特征（'mean'，'std'，'max'，'min'，'range'，'sum')，y_train是训练集的标签（'nb'，分类目标）

# 构建并训练随机森林模型
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 创建随机森林模型，n_estimators为100，random_state为42
model_rf.fit(X_train, y_train)  # 训练随机森林模型

# 构建并训练XGBoost模型
model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)  # 创建XGBoost模型，n_estimators为100，random_state为42
model_xgb.fit(X_train, y_train)  # 训练XGBoost模型

# 构建集成模型
ensemble_model = VotingClassifier(estimators=[('rf', model_rf), ('xgb', model_xgb)], voting='soft')  # 创建集成模型，包含随机森林模型和XGBoost模型
ensemble_model.fit(X_train, y_train)  # 训练集合模型，其中X_train是训练集的特征（'mean'，'std'，'max'，'min'，'range'，'sum'），y_train是训练集的标签（'nb'，分类目标）

# 创建滚动窗口时间序列数据
def create_sequences(scaled_data, seq_length):  # 输入参数：归一化后的数据，序列长度
    xs = []  # 初始化输入数据列表
    ys = []  # 初始化输出数据列表
    for i in range(len(scaled_data) - seq_length):  # 遍历数据集，每次取seq_length个数据作为一个序列
        x = scaled_data[i:i + seq_length]  # 取当前位置到当前位置+seq_length的数据作为一个序列（二维）
        y = scaled_data[i + seq_length]  # 取当前位置+seq_length的数据作为一个序列的输出（一维）
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)  # 返回两个数组

# 归一化数据
scaler = MinMaxScaler()  # 创建MinMaxScaler对象
scaled_data = scaler.fit_transform(df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']])  # 返回归一化后的数据，是一个二维数组

# 设置滑动窗口大小
window = 90  # 定义滑动窗口的大小为90天。

# 构造训练集和测试集
X, y = create_sequences(scaled_data, window)
print("Before split:")
print("X shape:", X.shape)
print("y shape:", y.shape)
# 构造训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42)  # 使用train_test_split函数将数据集X和标签y划分为训练集和测试集。test_size=0.1表示测试集占10%。random_state=42确保分割的可重复性。
print("After split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 定义模型构建函数
def create_lstm_model(neurons1=32, neurons2=16, dropout1=0.1, dropout2=0.1, optimizer='adam'):
    model = Sequential()  # 创建一个Sequential模型
    model.add(Input(shape=(window, X_train.shape[2])))  # 添加输入层，输入的形状是(window, X_train.shape[2])
    model.add(LSTM(neurons1, return_sequences=True))  # 添加第一层LSTM层，单元个数为neurons1，返回序列为True
    model.add(Dropout(dropout1))  # 添加Dropout层，丢弃概率为dropout1
    model.add(LSTM(neurons2, return_sequences=False))  # 添加第二层LSTM层，单元个数为neurons2，返回序列为False
    model.add(Dropout(dropout2))  # 添加Dropout层，丢弃概率为dropout2
    model.add(Dense(64, activation='relu'))  # 添加全连接层，输出维度为64，激活函数为relu
    model.add(Dense(6, activation='linear'))  # 添加全连接层，输出维度为6，激活函数为linear
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])  # 编译模型，损失函数为mse，优化器为optimizer
    return model

# 定义超参数范围搜索函数
def objective(trial):
    neurons1 = trial.suggest_int('neurons1', 16, 128)  # 搜索第一层LSTM单元个数
    neurons2 = trial.suggest_int('neurons2', 8, 64)  # 搜索第二层LSTM单元个数
    dropout1 = trial.suggest_float('dropout1', 0.0, 0.5)
    dropout2 = trial.suggest_float('dropout2', 0.0, 0.5)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'Nadam'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 200)

    regressor = KerasRegressor(model=create_lstm_model, epochs=epochs, batch_size=batch_size, verbose=0)

    def train_and_evaluate(X_train_cv, y_train_cv, X_val_cv, y_val_cv):
        regressor.fit(X_train_cv, y_train_cv)
        return regressor.score(X_val_cv, y_val_cv)

    with ThreadPoolExecutor(max_workers=3) as executor:  # 使用多线程并发执行训练和评估
        futures = [executor.submit(train_and_evaluate, X_train_cv, y_train_cv, X_val_cv, y_val_cv)
                   for X_train_cv, X_val_cv, y_train_cv, y_val_cv in
                   [train_test_split(X_train, y_train, test_size=0.2, random_state=np.random.randint(0, 1000)) for _ in
                    range(3)]]
        scores = [future.result() for future in futures]

    return np.mean(scores)

# 使用Optuna进行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 优化50次

trial = study.best_trial
print("Best trial:")
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

# 使用最佳参数训练最终模型
best_params = {k: v for k, v in trial.params.items() if k in create_lstm_model.__code__.co_varnames}
model_lstm = create_lstm_model(**best_params)

# 创建EarlyStopping回调
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model_lstm.fit(X_train, y_train,
                    epochs=trial.params['epochs'],
                    batch_size=trial.params['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# 准备模型输入数据
lstm_input = scaled_data[-window:].reshape(1, window, 6)

# 预测
pred_front = model_lstm.predict(lstm_input).reshape(-1)  # 预测前6个号码
pred_front_rescaled = scaler.inverse_transform([pred_front])[0]  # 将预测结果反归一化为原始数据范围
front_balls = list(map(int, np.round(pred_front_rescaled)))  # 将预测结果转换为整数列表
features = extract_features(pd.DataFrame([front_balls], columns=['n1', 'n2', 'n3', 'n4', 'n5', 'n6']))  # 提取特征, 注意这里的features是一个DataFrame，传入数据必须是列表套列的形式。
pred_nb = label_encoder.inverse_transform([ensemble_model.predict(features)[0]])[0]  # 预测蓝球的号码
ticket = front_balls + [int(pred_nb)]
print(ticket)

# 保存预测的号码
df1 = pd.read_csv('balls_data.csv')
num = [df1['num'].values.tolist()[-1]]
new_balls_list = num + ticket
new_balls_pd = pd.DataFrame([new_balls_list], columns=['num', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue1'])
old_prd_df = pd.read_csv('predict_lottery_double_balls_data.csv')
updated_df = pd.concat([old_prd_df, new_balls_pd], ignore_index=True)
updated_df.to_csv('predict_lottery_double_balls_data.csv', index=False)