import pandas as pd
import random


def create_random_image_data(num_images, start_date, end_date):
    """
    Creates a DataFrame with random image IDs and randomized created dates.

    Args:
        num_images (int): The number of images to generate.
        start_date (str): The start date for the date range (e.g., '2023-01-01').
        end_date (str): The end date for the date range (e.g., '2023-12-31').

    Returns:
        pandas.DataFrame: A DataFrame with 'image_id' and 'created_date' columns.
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    random_dates = [start_date + pd.Timedelta(days=random.randint(0, days_between_dates), seconds=random.randint(0, 86399)) for _ in range(num_images)] #86399 is the number of seconds in a day -1.
    data = {
        'image_id': range(num_images),
        'created_date': random_dates
    }
    return pd.DataFrame(data)


# Example usage:
num_images = 1000
start_date = '2025-01-01'
end_date = '2025-12-31'

df = create_random_image_data(num_images, start_date, end_date)

print(df)

# 1. Single day having maximum images
daily_counts = df['created_date'].dt.date.value_counts()
max_day = daily_counts.index[0]
max_day_count = daily_counts.iloc[0]

print(f"1. Single day with maximum images: {max_day} with {max_day_count} images")

# 2. Single week having maximum images
weekly_counts = df.groupby(pd.Grouper(key='created_date', freq='W'))['image_id'].count()
max_week = weekly_counts.idxmax()
max_week_count = weekly_counts.max()

print(f"2. Single week with maximum images: {max_week} with {max_week_count} images")

# 3. Single fortnight having maximum images
# Assuming fortnight is 14 days
fortnightly_counts = df.groupby(pd.Grouper(key='created_date', freq='14D'))['image_id'].count()
max_fortnight = fortnightly_counts.idxmax()
max_fortnight_count = fortnightly_counts.max()

print(f"3. Single fortnight with maximum images: {max_fortnight} with {max_fortnight_count} images")

# 4. Single month having maximum images
monthly_counts = df.groupby(pd.Grouper(key='created_date', freq='ME'))['image_id'].count()
max_month = monthly_counts.idxmax()
max_month_count = monthly_counts.max()

print(f"4. Single month with maximum images: {max_month} with {max_month_count} images")
