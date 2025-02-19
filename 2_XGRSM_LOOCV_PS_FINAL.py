import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, cross_validate, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
excel_file_path = r'C:\Users\Xiong Zhen\Desktop\phd\ML-EF\Normalized_MLDATA_AllSheets.xlsx'
sheet_name = 3
data = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# 定义输入和输出
X = data[['pH', 'potential (V)', 'K2SO4 conc (mM)', 'initial conc of pollutant (mg/L)', 'contact time (h)']]
output_columns = ['energy efficiency']

# 为每个输出变量建立独立模型
results = {}
figures = []  # 保存所有图像对象

for column in output_columns:
    O = data[column]

    # 定义XGBoost模型
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # 使用留一法进行模型评估
    def leave_one_out_model_evaluation(X, O, model):
        loo = LeaveOneOut()
        press_values = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            O_train, O_test = O.iloc[train_index], O.iloc[test_index]

            # 训练模型
            model.fit(X_train, O_train)

            # 预测并计算PRESS
            y_test_pred = model.predict(X_test)
            press_i = (O_test.values[0] - y_test_pred[0]) ** 2
            press_values.append(press_i)

        # 计算总PRESS和pred_r2
        total_press = np.sum(press_values)
        ss_total = np.sum((O - np.mean(O)) ** 2)
        pred_r2 = 1 - (total_press / ss_total)

        return pred_r2

    param_grid = {
        'learning_rate': [0.01, 0.25, 0.5],
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3],
        'subsample': [0.4, 0.8],
        'colsample_bytree': [0.4, 0.8],
        'reg_lambda': [0, 1],
        'reg_alpha': [0, 1]
    }

    # 初始化记录最佳模型的变量
    best_pred_r2 = float('-inf')
    best_params = None
    best_model = None
    best_r2 = float('-inf')

    # 遍历参数网格
    for learning_rate in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for reg_lambda in param_grid['reg_lambda']:
                            for reg_alpha in param_grid['reg_alpha']:
                                model.set_params(learning_rate=learning_rate, n_estimators=n_estimators,
                                                 max_depth=max_depth, subsample=subsample,
                                                 colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                                                 reg_alpha=reg_alpha)

                                # 使用新的评价方法计算pred_r2
                                pred_r2 = leave_one_out_model_evaluation(X, O, model)
                                print(f"Params: {learning_rate, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha}, Pred R2: {pred_r2:.4f}")

                                # 使用模型预测整个数据集
                                model.fit(X, O)
                                predictions = model.predict(X)
                                r2 = r2_score(O, predictions)

                                # 仅当整体R2大于0.6时更新最佳模型
                                if r2 >= 0.6 and pred_r2 > best_pred_r2:
                                    best_pred_r2 = pred_r2
                                    best_r2 = r2
                                    best_params = {'learning_rate': learning_rate, 'n_estimators': n_estimators,
                                                   'max_depth': max_depth, 'subsample': subsample,
                                                   'colsample_bytree': colsample_bytree, 'lambda': reg_lambda,
                                                   'alpha': reg_alpha}
                                    best_model = model

    # 如果找到符合条件的最佳模型，保存并绘制图表
    if best_model is not None:
        print(f"Best Model Parameters for {column}:", best_params)
        print(f"Best Model Pred R2 for {column}:", best_pred_r2)
        print(f"Best Model R2 for {column}:", best_r2)

        # 重新进行一次交叉验证来获取最佳模型的平均训练和测试R2分数
        kf = KFold(n_splits=5, shuffle=True, random_state=41)
        scores = cross_validate(best_model, X, O, cv=kf, scoring='r2', return_train_score=True)
        best_avg_train_score = np.mean(scores['train_score'])
        best_avg_test_score = np.mean(scores['test_score'])

        print(f"Best Model average train R2 for {column}:", best_avg_train_score)
        print(f"Best Model average test R2 for {column}:", best_avg_test_score)

        # 使用最佳模型在整个数据集上进行预测
        predictions = best_model.predict(X)

        # 打印最佳模型的性能指标
        mse = mean_squared_error(O, predictions)
        mae = mean_absolute_error(O, predictions)

        print(f"Best Model MSE for {column}:", mse)
        print(f"Best Model MAE for {column}:", mae)

        results[column] = {
            'best_params': best_params,
            'best_pred_r2': best_pred_r2,
            'best_avg_train_score': best_avg_train_score,
            'best_avg_test_score': best_avg_test_score,
            'mse': mse,
            'mae': mae,
            'r2': best_r2
        }

        # 保存模型
        best_model.save_model(f'2_ML_xgboost_model_{column}_s1s2avg_nor.xgb')

        # 保存图像对象到figures列表
        fig, ax = plt.subplots()
        ax.scatter(O, predictions, color='blue', alpha=0.5)
        ax.plot([min(O), max(O)], [min(O), max(O)], color='red')  # 画出y=x线
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs. Predicted for {column}')
        ax.text(min(O), max(predictions) * 0.8, f'R2: {best_r2:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        ax.grid(True)

        # 将图像对象添加到figures列表
        figures.append(fig)

    else:
        print(f"No model for {column} achieved R2 >= 0.6")

# 一次性输出所有图像
for fig in figures:
    plt.figure(fig.number)
    plt.show()

# 打印所有满足条件的模型的结果
for column, metrics in results.items():
    print(f"\nResults for {column}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
