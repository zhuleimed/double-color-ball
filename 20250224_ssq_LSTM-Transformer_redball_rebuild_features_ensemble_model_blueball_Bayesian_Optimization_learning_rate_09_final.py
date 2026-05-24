import numpy as np  # 导入NumPy库，用于数组和数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # 从Sklearn导入MinMaxScaler，用于数据归一化，LabelEncoder用于标签编码
from sklearn.model_selection import TimeSeriesSplit, train_test_split  # 从Sklearn导入TimeSeriesSplit，用于时间序列数据划分;导入train_test_split用于数据集划分
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # 从Sklearn导入RandomForestClassifier和VotingClassifier
from scikeras.wrappers import KerasRegressor  # 从Scikeras导入KerasRegressor，用于将Keras模型包装为Scikit-learn接口
from keras.callbacks import EarlyStopping  # 从Keras导入EarlyStopping，用于防止过拟合的早停机制
import xgboost as xgb  # 导入XGBoost库
import optuna  # 导入Optuna库，用于超参数优化
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error  # 导入评价指标函数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Embedding  # 导入Keras层
from tensorflow.keras.models import Model, Sequential  # 导入Keras模型和层
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam  # 导入Adam、RMSprop和Nadam优化器

# 读取数据
df = pd.read_csv('balls_data.csv').drop(columns=['num']).dropna().reset_index(drop=True)
# 从CSV文件读取数据，删除'num'列，丢弃缺失值，并重置索引
df.columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'nb']
# 重命名数据框的列名
print(df.shape)  # 打印数据框的形状（行数和列数）

# ================== 红球特征工程 ==================
def create_sequences(data, window_size):
    # 定义函数，用于创建时间序列窗口数据
    X, y = [], []  # 初始化输入特征X和目标特征y的列表
    for i in range(len(data) - window_size):
        # 遍历数据，生成以window_size为滑动窗口的序列
        X.append(data[i:i + window_size, :6])  # 添加窗口内的前6列作为输入特征
        y.append(data[i + window_size, :6])  # 添加窗口结束时的前6列作为目标特征
    return np.array(X), np.array(y)  # 返回特征和目标值的NumPy数组

# 精确归一化到[0,1]范围(1-33 → 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化MinMaxScaler，将数据缩放到[0,1]范围
scaled_red = scaler.fit_transform(df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values)
# 对红球特征（前6列）进行归一化

# 构造时间序列窗口
window = 90  # 定义滑动窗口的大小为90
X, y = create_sequences(scaled_red, window)  # 调用函数生成时间序列数据
print('X',X.shape)
print('y',y.shape)

# Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        # 调整key_dim为embed_dim，确保输出维度与输入维度一致
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]  # 确保ffn的输出维度与embed_dim相同
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # 调用MultiHeadAttention层
        attn_output = self.att(inputs, inputs)
        # 对注意力输出进行dropout操作
        attn_output = self.dropout1(attn_output, training=training)
        # 残差连接并进行层归一化
        out1 = self.layernorm1(inputs + attn_output)
        # 前馈神经网络
        ffn_output = self.ffn(out1)
        # 对前馈神经网络输出进行dropout操作
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差连接并进行层归一化
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape


# 定义LSTM-Transformer模型
def create_lstm_transformer_model(neurons1=32, neurons2=16, dropout1=0.1, transformer_dim=64, num_heads=4, ff_dim=128,
                                  optimizer='adam', learning_rate=0.001):
    inputs = Input(shape=(window, 6))
    x = LSTM(neurons1, return_sequences=True)(inputs)
    x = Dropout(dropout1)(x)

    # Transformer模块
    # 确保TransformerBlock的embed_dim与LSTM的输出维度一致
    transformer = TransformerBlock(embed_dim=neurons1, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer(x)

    x = LSTM(neurons2, return_sequences=False)(x)
    x = Dropout(dropout1)(x)

    outputs = Dense(6, activation='linear')(x)

    model = Model(inputs, outputs)

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=learning_rate)

    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    return model

# 超参数搜索函数
def objective(trial):
    neurons1 = trial.suggest_int('neurons1', 16, 128)
    neurons2 = trial.suggest_int('neurons2', 8, 64)
    dropout1 = trial.suggest_float('dropout1', 0.0, 0.5)
    transformer_dim = trial.suggest_int('transformer_dim', 32, 128)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    ff_dim = trial.suggest_int('ff_dim', 64, 256)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'Nadam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 200)

    regressor = KerasRegressor(
        model=create_lstm_transformer_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        neurons1=neurons1,
        neurons2=neurons2,
        dropout1=dropout1,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        regressor.fit(X_train, y_train)
        scores.append(regressor.score(X_val, y_val))

    return np.mean(scores)

# 运行优化（最小化MSE）
study = optuna.create_study(direction='minimize')  # 创建Optuna研究，优化方向为最小化
study.optimize(objective, n_trials=20, n_jobs=4)  # 优化目标函数，进行20次试验，使用4个并行作业

best_trial = study.best_trial  # 获取最佳试验
print("Best trial:")  # 打印最佳试验信息
print(f"Value: {best_trial.value}")  # 打印最佳试验的目标值
print("Params: ")
for key, value in best_trial.params.items():
    # 打印最佳试验的超参数
    print(f"{key}: {value}")

# 使用最佳参数训练最终模型
best_params = {k: v for k, v in best_trial.params.items() if k in create_lstm_transformer_model.__code__.co_varnames}
'''

best_params = {}
best_params['neurons1'] = 64
best_params['neurons2'] = 45
best_params['dropout1'] = 0.05691807189189449
best_params['dropout2'] = 0.46186025273773645
best_params['optimizer'] = 'rmsprop'
'''

# 提取与模型构建函数参数匹配的最佳超参数
model_lstm_transformer = create_lstm_transformer_model(**best_params)  # 使用最佳参数构建LSTM模型

# 准备训练数据（前90%训练，后10%测试）
train_size = int(len(X) * 0.9)  # 计算训练集大小
X_train, X_test = X[:train_size], X[train_size:]  # 划分训练集和测试集
y_train, y_test = y[:train_size], y[train_size:]

print(X_train.shape, y_train.shape)
print('X_test', X_test.shape)

# 创建EarlyStopping回调
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 定义早停机制，监控验证损失，如果10轮未改善则停止训练，并恢复最佳权重

# 训练模型
history = model_lstm_transformer.fit(X_train, y_train,
                         epochs=best_trial.params['epochs'],  # 使用最佳轮数
                         # epochs=50,
                         batch_size=best_trial.params['batch_size'],  # 使用最佳批量大小
                         # batch_size=64,
                         validation_data=(X_test, y_test),  # 使用测试集作为验证集
                         callbacks=[early_stopping],  # 使用早停回调
                         verbose=1)  # 打印训练过程日志

# 评价LSTM模型在测试集上的表现

y_pred_lstm = model_lstm_transformer.predict(X_test)
print('y_pred_lstm', y_pred_lstm)
y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
y_test_rescaled = scaler.inverse_transform(y_test)
print("y_pred_lstm_rescaled:", y_pred_lstm_rescaled)
print('y_test_rescaled:', y_test_rescaled)
exit()
# ================== 兰球特征工程 ==================
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
model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)# 创建XGBoost模型，n_estimators为100，random_state为42
model_xgb.fit(X_train, y_train)  # 训练XGBoost模型

# 构建集成模型
ensemble_model = VotingClassifier(estimators=[('rf', model_rf), ('xgb', model_xgb)], voting='soft')  # 创建集成模型，包含随机森林模型和XGBoost模型
ensemble_model.fit(X_train, y_train)  # 训练集合模型，其中X_train是训练集的特征（'mean'，'std'，'max'，'min'，'range'，'sum'），y_train是训练集的标签（'nb'，分类目标）

# 提取特征
features = extract_features(pd.DataFrame([front_balls], columns=['n1', 'n2', 'n3', 'n4', 'n5', 'n6']))
# 归一化处理
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)

# 转换为 pandas.DataFrame 并指定列名
features_scaled_df = pd.DataFrame(features_scaled, columns=['mean', 'std', 'max', 'min', 'range', 'sum'])  # 将特征转换为pandas.DataFrame，并指定和训练时一致的列名。

# 预测兰球
pred_nb = label_encoder.inverse_transform([ensemble_model.predict(features_scaled_df)[0]])[0]
ticket = front_balls + [int(pred_nb)]
print(ticket)

# 保存预测的号码
df1 = pd.read_csv('balls_data.csv')
num = [df1['num'].values.tolist()[-1]]
num = num[0] + 1
new_balls_list = [num] + ticket
new_balls_pd = pd.DataFrame([new_balls_list], columns=['num', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue1'])
old_prd_df = pd.read_csv('predict_lottery_double_balls_data.csv')
updated_df = pd.concat([old_prd_df, new_balls_pd], ignore_index=True)
updated_df.to_csv('predict_lottery_double_balls_data.csv', index=False)