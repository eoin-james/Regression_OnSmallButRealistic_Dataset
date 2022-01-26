import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
SEED = 1234
np.random.seed(SEED)

cols = ['alcohol_by_weight', 'rating', 'bitterness', 'nitrogen', 'turbidity', 'sugars', 'degree_of_fermentation',
        'calorific_value', 'density', 'pH', 'colour', 'sulphites']

file_path = 'BeerData/beer_rating.csv'


def prepro(seed):
    data_df = pd.read_csv(file_path, sep='\t', engine='python', header=None)
    data_df.columns = cols

    x_data = data_df.iloc[:, data_df.columns != "rating"]
    # x_data = x_data.iloc[:, x_data.columns != "bitterness"]
    # x_data = x_data.iloc[:, x_data.columns != "density"]
    x_data = np.asarray(x_data)
    y_data = np.array(data_df.iloc[:, data_df.columns == "rating"])

    return train_test_split(x_data, y_data, train_size=0.70, shuffle=True, random_state=seed)


def linear_reg(x_train_lr, x_test_lr, y_train_lr, y_test_lr, seed_lr):
    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(x_train_lr, y_train_lr)
    y_pred = lin_reg.predict(x_test_lr)
    results(lin_reg, y_test_lr, y_pred, 'Linear Regressor', seed_lr)


def MLPR(x_train, x_test, y_train, y_test, seed_mlpr):
    mlpr = MLPRegressor(random_state=None, max_iter=10_000)
    mlpr.fit(x_train, y_train.ravel())
    y_pred = mlpr.predict(x_test)
    results(mlpr, y_test, y_pred, 'Multi-Layer Perceptron Regressor', seed_mlpr)


def results(model_r, y_test_r, y_pred_, model_string, seed_r):
    mse = mean_squared_error(y_test_r, y_pred_, squared=True)
    rmse = mean_squared_error(y_test_r, y_pred_, squared=False)
    model_score = model_r.score(x_test, y_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_)

    print(f'Model Score: {model_score}'.format(model_score))
    print(f'Mean squared error: {mse}')
    print(f'Root mean squared error: {rmse}')
    print(f'Mean absolute error: {mae}')
    print('\n')

    y_test_r = y_test_r.flatten()
    y_pred_ = y_pred_.flatten()

    x = np.arange(1, len(y_test_r) + 1)
    plt.scatter(x[:20], y_test_r[:20], c='#3F9F4C', s=20, label='Test Values')
    plt.scatter(x[:20], y_pred_[:20], c='#FF0000', s=20, label='Predicted Values')
    plt.xlabel('Observation')
    plt.ylabel('Rating')
    plt.title(f'Model: {model_string} - Seed: {seed_r}')
    plt.legend()
    plt.savefig(f'{model_string}_mini_results_{seed_r}')
    plt.show()

    plt.scatter(x, y_test_r, c='#3F9F4C', s=15, label='Test Values')
    plt.scatter(x, y_pred_, c='#FF0000', s=15, label='Predicted Values')
    plt.xlabel('Observation')
    plt.ylabel('Rating')
    plt.title(f'Model: {model_string} - Seed: {seed_r}')
    plt.legend()
    plt.savefig(f'{model_string}_results_{seed_r}')
    plt.show()


LR = True
MLP = True
res = False
if __name__ == '__main__':
    seeds = np.random.randint(10_000, size=5)
    for seed in seeds:
        x_train, x_test, y_train, y_test = prepro(seed)
        if LR:
            print('Linear Regression - Seed:', seed)
            linear_reg(x_train, x_test, y_train, y_test, seed)
        if MLP:
            print('Multi Layer Perceptron - Seed:', seed)
            MLPR(x_train, x_test, y_train, y_test, seed)

        if res:
            mlpr = MLPRegressor(random_state=None, max_iter=10_000)
            mlpr.fit(x_train, y_train.ravel())
            y_pred_mlp = mlpr.predict(x_test)

            lin_reg = LinearRegression(normalize=True)
            lin_reg.fit(x_train, y_train)
            y_pred_lr = lin_reg.predict(x_test)

            x = np.arange(1, len(y_test) + 1)
            plt.scatter(x[:20], y_test[:20], c='#3F9F4C', s=20, label='Test Values')
            plt.scatter(x[:20], y_pred_mlp[:20], c='#1035FF', s=20, label='MLP Predicted Values')
            plt.scatter(x[:20], y_pred_lr[:20], c='#E32FFF', s=20, label='LR Predicted Values')
            plt.xlabel('Observation')
            plt.ylabel('Rating')
            plt.title(f'Model: MLP v LR - Seed: {seed}')
            plt.legend()
            plt.savefig(f'MLPvLR_mini_results_{seed}')
            plt.show()

            plt.scatter(x, y_test, c='#3F9F4C', s=20, label='Test Values')
            plt.scatter(x, y_pred_mlp, c='#1035FF', s=20, label='MLP Predicted Values')
            plt.scatter(x, y_pred_lr, c='#E32FFF', s=20, label='LR Predicted Values')
            plt.xlabel('Observation')
            plt.ylabel('Rating')
            plt.title(f'Model: MLP v LR - Seed: {seed}')
            plt.legend()
            plt.savefig(f'MLPvLR_results_{seed}')
            plt.show()