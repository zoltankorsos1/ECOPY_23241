import pandas as pd

import pandas as pd
import os

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

def items_and_prices(input_df):
    result_df = input_df[['item_name', 'item_price']].copy()

    return result_df

def avg_price(input_df):
    average_price = input_df['item_price'].mean()

    return average_price

def unique_items_over_ten_dollars(input_df):
    filtered_df = input_df[input_df['item_price'] > 10]
    filtered_df = filtered_df[['item_name', 'choice_description', 'item_price']]
    unique_items_df = filtered_df.drop_duplicates()

    return unique_items_df

def items_starting_with_s(input_df):
    filtered_df = input_df[input_df['item_name'].str.startswith('S')]
    filtered_df = filtered_df[['item_name']]

    return filtered_df

def first_three_columns(input_df):
    selected_columns = input_df.iloc[:, :3]
    return selected_columns


def every_column_except_last_two(input_df):
    selected_columns = input_df.iloc[:, :-2]
    return selected_columns


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_df[columns_to_keep]


def generate_quarters(input_df):
    def quartile(price):
        if price <=30:
            return 'premium'
        elif 20 <= price <= 29.99:
            return 'high-cost'
        elif 10 <= price <= 19.99:
            return 'medium-cost'
        else:
            return 'low-cost'

    input_df['Quartile'] = input_df['Price'].apply(quartile)
    return input_df

def average_price_in_quartiles(input_df):
    avg_price_in_quartiles = input_df.groupby('Quartile')['item_price'].mean()
    return avg_price_in_quartiles


def minmaxmean_price_in_quartile(input_df):
    minmaxmean_price = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])
    return minmaxmean_price

from random import seed, uniform
def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Inicializáljuk a random generátort a megadott seed értékkel (42)
    seed(42)

    # Lista a kimeneti trajectóriák tárolására
    trajectories = []

    for _ in range(number_of_trajectories):
        # Kezdeti érték az adott trajectória kumulatív átlagához
        cumulative_average = 0.0

        # Trajectória létrehozása
        trajectory = []

        for _ in range(length_of_trajectory):
            # Véletlenszerűen generált szám uniform eloszlás alapján
            random_value = distribution.uniform(0, 1)

            # Kumulatív átlag frissítése
            cumulative_average = (cumulative_average + random_value) / (len(trajectory) + 1)

            # Hozzáadjuk a kumulatív átlagot a trajectóriához
            trajectory.append(cumulative_average)

        # Hozzáadjuk a trajectóriát a kimeneti listához
        trajectories.append(trajectory)

    return trajectories


from random import seed, gauss

def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Inicializáljuk a random generátort a megadott seed értékkel (42)
    seed(42)

    # Lista a kimeneti trajectóriák tárolására
    trajectories = []

    for _ in range(number_of_trajectories):
        # Kezdeti érték az adott trajectória kumulatív átlagához
        cumulative_average = 0.0

        # Trajectória létrehozása
        trajectory = []

        for _ in range(length_of_trajectory):
            # Véletlenszerűen generált szám Gauss-eloszlás alapján
            random_value = distribution.gauss(1, 3.3)

            cumulative_average = (cumulative_average + random_value) / (len(trajectory) + 1)

            trajectory.append(cumulative_average)

        # Hozzáadjuk a trajectóriát a kimeneti listához
        trajectories.append(trajectory)

    return trajectories


from random import seed, gauss

def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Inicializáljuk a random generátort a megadott seed értékkel (42)
    seed(42)

    # Lista a kimeneti trajectóriák tárolására
    trajectories = []

    for _ in range(number_of_trajectories):
        # Kezdeti érték az adott trajectória kumulatív átlagához
        cumulative_average = 0.0

        # Trajectória létrehozása
        trajectory = []

        for _ in range(length_of_trajectory):
            # Véletlenszerűen generált szám Gauss-eloszlás alapján
            random_value = distribution.gauss(1, 3.3)

            # Kumulatív átlag frissítése
            cumulative_average = (cumulative_average + random_value) / (len(trajectory) + 1)

            # Hozzáadjuk a kumulatív átlagot a trajectóriához
            trajectory.append(cumulative_average)

        # Hozzáadjuk a trajectóriát a kimeneti listához
        trajectories.append(trajectory)

    return trajectories

from random import seed, cauchy

def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Inicializáljuk a random generátort a megadott seed értékkel (42)
    seed(42)

    # Lista a kimeneti trajectóriák tárolására
    trajectories = []

    for _ in range(number_of_trajectories):
        # Kezdeti érték az adott trajectória kumulatív átlagához
        cumulative_average = 0.0

        # Trajectória létrehozása
        trajectory = []

        for _ in range(length_of_trajectory):
            # Véletlenszerűen generált szám Cauchy-eloszlás alapján
            random_value = distribution(2, 4)

            # Kumulatív átlag frissítése
            cumulative_average = (cumulative_average + random_value) / (len(trajectory) + 1)

            # Hozzáadjuk a kumulatív átlagot a trajectóriához
            trajectory.append(cumulative_average)

        # Hozzáadjuk a trajectóriát a kimeneti listához
        trajectories.append(trajectory)

    return trajectories

from random import seed, chisquare

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Inicializáljuk a random generátort a megadott seed értékkel (42)
    seed(42)

    # Lista a kimeneti trajectóriák tárolására
    trajectories = []

    for _ in range(number_of_trajectories):
        # Kezdeti érték az adott trajectória kumulatív átlagához
        cumulative_average = 0.0

        # Trajectória létrehozása
        trajectory = []

        for _ in range(length_of_trajectory):
            # Véletlenszerűen generált szám Chi-squared eloszlás alapján
            random_value = distribution(3)

            # Kumulatív átlag frissítése
            cumulative_average = (cumulative_average + random_value) / (len(trajectory) + 1)

            # Hozzáadjuk a kumulatív átlagot a trajectóriához
            trajectory.append(cumulative_average)

        # Hozzáadjuk a trajectóriát a kimeneti listához
        trajectories.append(trajectory)

    return trajectories







