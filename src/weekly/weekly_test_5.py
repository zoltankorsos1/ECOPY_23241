import pandas as pd

import pandas as pd
import random


def change_price_to_float(input_df):
    modified_df = input_df.copy()
    modified_df["item_price"] = modified_df["item_price"].str.replace('$', '').astype(float)

    return modified_df

def number_of_observations(input_df):
    num_observations = len(input_df)
    return num_observations


def items_and_prices(input_df):
    result_df = input_df[['item_name', 'item_price']].copy()

    return result_df


def sorted_by_price(input_df):
    items_prices_df = items_and_prices(input_df)
    sorted_df = items_prices_df.sort_values(by='item_price', ascending=False)

    return sorted_df

def avg_price(input_df):
    average_price = input_df['item_price'].mean()

    return average_price

def unique_items_over_ten_dollars(input_df):
    filtered_df = input_df[input_df['item_price'] > 10]
    filtered_df = filtered_df[['item_name', 'choice_description', 'item_price']]
    unique_items_df = filtered_df.drop_duplicates()

    return unique_items_df

def items_starting_with_s(input_df):
    filtered_df = input_df['item_name'][input_df['item_name'].str.startswith('S')]
    groups = filtered_df.drop_duplicates()
    return groups


def first_three_columns(input_df):
    selected_columns = input_df.iloc[:, :3]
    return selected_columns


def every_column_except_last_two(input_df):
    selected_columns = input_df.iloc[:, :-2]
    return selected_columns


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_df[columns_to_keep]


def generate_quartile(input_df):
    input_df = input_df.copy()

    def quartile(pre):
        if 30 < pre:
            return "premium"
        elif 20 <= pre <= 29.99:
            return "high-cost"
        elif 10 <= pre <= 19.99:
            return "medium-cost"
        elif 0 <= pre <= 9.99:
            return 'low-cost'
    input_df['Quartile'] = input_df['item_price'].apply(quartile)
    return input_df
def average_price_in_quartiles(input_df):
    avg_price_in_quartiles = input_df.groupby('Quartile')['item_price'].mean()
    return avg_price_in_quartiles


def minmaxmean_price_in_quartile(input_df):
    minmaxmean_price = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])
    return minmaxmean_price


def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_rand()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result


def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_rand()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result


def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_random()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result


def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.generate_random()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result


def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_mean = 0.0
        for _ in range(length_of_trajectory):
            value = distribution.gen_rand()
            cumulative_mean += value
            trajectory.append(cumulative_mean / (len(trajectory) + 1))
        result.append(trajectory)
    return result









