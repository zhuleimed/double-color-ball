# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import threading
import optuna
import os

# 全局锁
eval_lock = threading.Lock()
pred_lock = threading.Lock()

# 全局初始化组件
blue_encoder = LabelEncoder()  # 蓝球编码器（线程安全）


# -------------------- 核心函数 --------------------
def ssq_predict():
    try:
        # ================== 数据读取与预处理 ==================
        df = pd.read_csv('balls_data.csv')
        df = df.drop(columns=['num']).dropna().reset_index(drop=True)
        df.columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'nb']

        # 全局初始化蓝球编码器
        global blue_encoder
        blue_encoder.fit(df['nb'])  # 保证所有线程使用同一编码器

        # ================== 红球特征工程 ==================
        def create_sequences(data, window_size):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i + window_size, :6])  # 窗口数据
                y.append(data[i + window_size, :6])  # 下一期红球
            return np.array(X), np.array(y)

        # 精确归一化到[0,1]范围(1-33 → 0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_red = scaler.fit_transform(df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values)

        # 构造时间序列窗口
        window = 90
        X, y = create_sequences(scaled_red, window)

        # 按时间顺序划分数据集
        train_size = int(len(X) * 0.9)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # ================== LSTM模型构建 ==================
        def create_lstm_model(neurons1=64, neurons2=32, dropout=0.2, lr=1e-3, dense_units=64, use_dense=True):
            model = Sequential([
                Input(shape=(window, 6)),
                LSTM(neurons1, return_sequences=True),
                Dropout(dropout),
                BatchNormalization(),
                LSTM(neurons2),
                Dropout(dropout),
                BatchNormalization()
            ])

            if use_dense:
                model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4)))
                model.add(Dropout(dropout))
                model.add(BatchNormalization())

            # 统一添加输出层（无论是否使用全连接）
            model.add(Dense(6, activation='sigmoid'))

            # 统一编译模型
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='mse',
                metrics=['mae'],
                jit_compile=True
            )
            return model

        # ================== Optuna超参数优化 ==================
        def objective(trial):
            use_dense = trial.suggest_categorical('use_dense', [True, False])

            params = {
                'neurons1': trial.suggest_int('neurons1', 64, 256),
                'neurons2': trial.suggest_int('neurons2', 32, 128),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'use_dense': use_dense
            }

            # 仅当使用全连接层时添加相关参数
            if use_dense:
                params['dense_units'] = trial.suggest_int('dense_units', 32, 128)

            model = KerasRegressor(
                model=create_lstm_model,
                **params,
                epochs=100,
                verbose=0
            )

            # 学习率调度
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

            # 时间序列交叉验证
            scores = []
            tscv = TimeSeriesSplit(n_splits=3)
            for train_idx, val_idx in tscv.split(X_train):
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    validation_data=(X_train[val_idx], y_train[val_idx]),
                    callbacks=[reduce_lr],
                    verbose=0
                )
                pred = model.predict(X_train[val_idx])
                scores.append(mean_squared_error(y_train[val_idx], pred))
            return np.mean(scores)

        # 运行优化（最小化MSE）
        study = optuna.create_study(direction='minimize')  # 最小化方向优化MSE，适合离散值预测的误差最小化需求
        study.optimize(objective, n_trials=30, n_jobs=2)

        # ================== 训练最终模型 ==================
        best_params = study.best_params
        model = create_lstm_model(**best_params)

        # 回调函数
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=best_params['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # ================== 红球预测 ==================
        last_window = scaled_red[-window:].reshape(1, window, 6)
        pred_red_scaled = model.predict(last_window)[0]

        # 精确反归一化 (0-1 → 1-33)
        pred_red = (pred_red_scaled * 32 + 1).astype(int)
        pred_red = np.clip(pred_red, 1, 33)

        # 去重并确保6个号码
        pred_red = np.unique(pred_red)
        if len(pred_red) < 6:
            # 补充随机数（简单处理，可改进）
            supplement = np.random.choice(np.setdiff1d(np.arange(1, 34), pred_red), 6 - len(pred_red), replace=False)
            pred_red = np.concatenate([pred_red, supplement])
        pred_red = np.sort(pred_red[:6])  # 取前6个并排序

        # ================== 蓝球预测 ==================
        # 计算统计特征
        features = pd.DataFrame({
            'mean': df.iloc[:, :6].mean(axis=1),
            'std': df.iloc[:, :6].std(axis=1),
            'max': df.iloc[:, :6].max(axis=1),
            'min': df.iloc[:, :6].min(axis=1),
            'range': df.iloc[:, :6].max(axis=1) - df.iloc[:, :6].min(axis=1),
            'sum': df.iloc[:, :6].sum(axis=1)
        })

        # 训练集成模型
        X_blue = features.values
        y_blue = blue_encoder.transform(df['nb'])  # 使用全局编码器

        model_rf = RandomForestClassifier(n_estimators=100)
        model_xgb = xgb.XGBClassifier(n_estimators=100)
        ensemble = VotingClassifier(estimators=[('rf', model_rf), ('xgb', model_xgb)], voting='soft')
        ensemble.fit(X_blue, y_blue)  # 使用全量数据训练（假设数据量足够）

        # 预测蓝球
        current_feature = [[
            pred_red.mean(), pred_red.std(),
            pred_red.max(), pred_red.min(),
            pred_red.ptp(), pred_red.sum()
        ]]
        pred_blue_encoded = ensemble.predict(current_feature)
        pred_blue = blue_encoder.inverse_transform(pred_blue_encoded)[0]

        # ================== 结果保存 ==================
        result = {
            'red': '-'.join(map(str, pred_red)),
            'blue': pred_blue
        }

        with pred_lock:
            # 原子化写入操作
            file_exists = os.path.exists('ssq_predictions.csv')
            pd.DataFrame([result]).to_csv('ssq_predictions.csv', mode='a', header=not file_exists, index=False)

    except Exception as e:
        print(f'预测出错: {e}')


# -------------------- 多线程执行 --------------------
def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ssq_predict) for _ in range(5)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()