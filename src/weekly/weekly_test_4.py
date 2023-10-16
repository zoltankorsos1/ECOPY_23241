from pytest import approx
import pandas as pd
import random
from pandas.testing import assert_frame_equal, assert_series_equal
import src.weekly.weekly_test_4 as wt
from src.weekly.weekly_test_2 import ParetoDistribution

euro12 = pd.read_csv('../data/Euro_2012_stats_TEAM.csv')

def number_of_participants(input_df):
    # Megszámoljuk a sorokat a DataFrame-ben, ami megadja a résztvevő csapatok számát
    num_participants = len(input_df)
    return num_participants


def goals(input_df):
    # Ellenőrizzük, hogy az 'input_df' nevű DataFrame tartalmazza-e a 'Team' és 'Goals' oszlopokat.
    if 'Team' in input_df.columns and 'Goals' in input_df.columns:
        # Kiválasztjuk csak a 'Team' és 'Goals' oszlopokat, majd visszaadjuk egy új DataFrame-ben.
        selected_columns = input_df[['Team', 'Goals']]
        return selected_columns
    else:
        return pd.DataFrame()  # Ha nincsenek megfelelő oszlopok, üres DataFrame-et adunk vissza.


def sorted_by_goal(input_df):
    goal_data = goals(input_df)
    sorted_data = goal_data.sort_values(by='Goals', ascending=False)

    return sorted_data

def avg_goal(input_df):
    if 'Goals' in input_df.columns:
        total_goals = input_df['Goals'].sum()
        num_teams = len(input_df)
        avg = total_goals / num_teams
        return avg
    else:
        return 0

def countries_over_six(input_df):
    if 'Team' in input_df.columns and 'Goals' in input_df.columns:
        filtered_data = input_df[input_df['Goals'] >= 6]
        return filtered_data
    else:
        return pd.DataFrame()



def countries_starting_with_g(input_df):
    if 'Team' in input_df.columns:
        filtered_data = input_df[input_df['Team'].str.startswith('G')]
        return filtered_data
    else:
        return pd.DataFrame()

def first_seven_columns(input_df):
    if len(input_df.columns) >= 7:
        selected_columns = input_df.iloc[:, :7]
        return selected_columns
    else:
        return pd.DataFrame()

def every_column_except_last_three(input_df):
    if len(input_df.columns) > 3:
        selected_columns = input_df.iloc[:, :-3]
        return selected_columns
    else:
        return pd.DataFrame()


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    if all(col in input_df.columns for col in columns_to_keep) and column_to_filter in input_df.columns:
        selected_columns = input_df[columns_to_keep]
        filtered_rows = input_df[input_df[column_to_filter].isin(rows_to_keep)]

        return selected_columns.merge(filtered_rows, on=columns_to_keep)
    else:
        return pd.DataFrame()


def generate_quartile(input_df):
    # Új 'Quartile' oszlop létrehozása a megfelelő értékekkel
    input_df['Quartile'] = pd.cut(input_df['Goals'], [-1, 2, 4, 5, 12], labels=[4, 3, 2, 1])

    return input_df


def average_yellow_in_quartiles(input_df):
    # Csoportosítás a 'Quartile' oszlop alapján és átlagos passzok kiszámítása
    quartile_avg_passes = input_df.groupby('Quartile')['Passes'].mean().reset_index()
    quartile_avg_passes.columns = ['Quartile', 'Average_Passes']

    return quartile_avg_passes


def minmax_block_in_quartile(input_df):
    # Csoportosítás a 'Quartile' oszlop alapján és blokkok minimális és maximális értékének kiszámítása
    quartile_minmax_blocks = input_df.groupby('Quartile')['Blocks'].agg(['min', 'max']).reset_index()
    quartile_minmax_blocks.columns = ['Quartile', 'Min_Blocks', 'Max_Blocks']

    return quartile_minmax_blocks


import matplotlib.pyplot as plt
def scatter_goals_shots(input_df):
    # Scatter plot létrehozása
    fig, ax = plt.subplots()
    ax.scatter(input_df['Goals'], input_df['Shots on target'])

    # Tengelyfeliratok beállítása
    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')

    # Cím beállítása
    ax.set_title('Goals and Shot on target')

    plt.show()

    return fig
# Tesztelés
scatter_plot = scatter_goals_shots(euro12)


import matplotlib.pyplot as plt

def scatter_goals_shots_by_quartile(input_df):
    grouped = input_df.groupby('Quartile')

    colors = ['b', 'g', 'r', 'c']

    legend_labels = []

    fig, ax = plt.subplots()
    for quartile, group in grouped:
        ax.scatter(group['Goals'], group['Shots on target'], label=f'Quartile {quartile}', color=colors[quartile - 1])
        legend_labels.append(f'Quartile {quartile}')

    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')

    ax.set_title('Goals and Shot on target')

    ax.legend(legend_labels, title='Quartiles')

    plt.show()

    return fig



from scipy.stats import pareto
import numpy as np
from typing import List


def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    np.random.seed(42)

    result = []

    for _ in range(number_of_trajectories):
        random_numbers = pareto_distribution.rvs(size=length_of_trajectory)

        cumulative_means = np.cumsum(random_numbers) / (np.arange(1, length_of_trajectory + 1))

        result.append(cumulative_means.tolist())

    return result


# %%
# Tesztelés
pareto_distribution = pareto(1, scale=1)  # Pareto eloszlás (1, 1) paraméterekkel
number_of_trajectories = 5
length_of_trajectory = 10
trajectories = gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory)
print(trajectories)












