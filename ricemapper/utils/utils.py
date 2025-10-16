# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime, timedelta
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

presto_dir = Path(__file__).parent.parent / "models" / "presto"
presto_dir.mkdir(exist_ok=True, parents=True)
print("Using presto dir: %s" % presto_dir)
presto_model_path = presto_dir / "default_model.pt"


def multipolygon_to_featurecollection(multipolygon):
    """
    Convert an ee.Geometry.MultiPolygon to an ee.FeatureCollection where each feature
    is one of the constituent polygons.

    Parameters:
    multipolygon (ee.Geometry.MultiPolygon): Input multipolygon

    Returns:
    ee.FeatureCollection: Collection where each feature contains one polygon
    """
    # Get the coordinates of the multipolygon
    coords = multipolygon.coordinates()

    # Function to convert each polygon coordinates to a feature
    def coords_to_feature(coords):
        return ee.Feature(ee.Geometry.Polygon(coords))

    # Map over the coordinates to create features
    features = ee.List(coords).map(lambda poly: coords_to_feature(poly))

    # Convert to FeatureCollection
    return ee.FeatureCollection(features)


def gdf_to_single_ee_geometry(gdf, buffer=None):
    """
    Convert a GeoDataFrame to a single ee.Geometry keeping only the exterior boundary.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame containing polygon geometries
    buffer (int): Buffer distance in meters to apply to the geometry

    Returns:
    ee.Geometry: Earth Engine geometry containing only the exterior boundary
    """
    # Combine all geometries into a single geometry
    combined = unary_union(gdf.geometry.values)

    if buffer is not None:
        combined = combined.buffer(buffer)

    # Extract exterior coordinates based on geometry type
    if isinstance(combined, MultiPolygon):
        # If still a MultiPolygon after union, get the largest polygon
        largest_poly = max(combined.geoms, key=lambda x: x.area)
        coords = list(largest_poly.exterior.coords)
    elif isinstance(combined, Polygon):
        coords = list(combined.exterior.coords)
    else:
        raise ValueError(f"Unexpected geometry type after union: {type(combined)}")

    # Convert coordinates to the format expected by Earth Engine
    ee_coords = [[coord[0], coord[1]] for coord in coords]

    # Create Earth Engine geometry
    return ee.Geometry.Polygon(ee_coords)


def find_gdf_ROI(gdf, buffer=500):  # buffer in meters
    bounds = gdf.total_bounds  # returns (minx, miny, maxx, maxy)

    # Calculate the center of the bounding box
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    ROI = ee.Geometry.Polygon(
        [
            [bounds[0], bounds[1]],  # (minx, miny)
            [bounds[0], bounds[3]],  # (minx, maxy)
            [bounds[2], bounds[3]],  # (maxx, maxy)
            [bounds[2], bounds[1]],  # (maxx, miny)
            [bounds[0], bounds[1]],  # (minx, miny) to close the polygon
        ],
        proj="EPSG:4326",
    )
    ROI = ROI.buffer(buffer).bounds()

    print("EE ROI: ", ROI.getInfo())

    return ROI


def align_dates_to_week_middle(timeseries_df, date_col="Dates", align_start_date=True):
    """
    Function to align the columns 'Dates' to the middle of the week.
    """

    # Function to convert a date to the middle of the week
    def to_middle_of_week(date):
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        start_of_week = date - timedelta(days=date.weekday())
        middle_of_week = start_of_week + timedelta(days=3)
        return middle_of_week.strftime("%Y-%m-%d")

    # Iterate over each row in the DataFrame
    for index, row in timeseries_df.iterrows():
        # Convert each date in the 'Dates' list to the middle of the week
        new_dates = [to_middle_of_week(date) for date in row[date_col]]
        # Save the new list of dates back to the DataFrame
        timeseries_df.at[index, date_col] = new_dates

        if align_start_date:
            timeseries_df.at[index, "Start_Date"] = to_middle_of_week(row["Start_Date"])

    return timeseries_df


# find number of weeks since 'Start_Date'
def find_weeks_since_start_date(timeseries_df):
    """
    Function to find the number of weeks since the 'Start_Date' in each row.
    """

    # create a new column called 'Weeks'
    timeseries_df["Weeks"] = None

    # Function to find the number of weeks since the 'Start_Date'
    def weeks_since_start_date(dates, start_date):

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        # else: #convert datetime to Y-m-d datetime
        #     start_date = start_date.strftime('%Y-%m-%d')

        return [(date - start_date).days // 7 for date in dates]

    # Iterate over each row in the DataFrame
    for index, row in timeseries_df.iterrows():
        # Find the number of weeks since the 'Start_Date'
        start_date = row["Start_Date"]
        # convert to string
        # start_date = str(start_date)
        weeks = weeks_since_start_date(row["Dates"], start_date)
        # Save the list of weeks back to the DataFrame
        timeseries_df.at[index, "Weeks"] = weeks

    return timeseries_df


def drop_all_negative_weeks(timeseries_df):
    """
    iterate over 'Timeseries', 'Dates' columns and remove all list elements that are before the first positive index of 'Weeks' column

    """

    # Function to drop all negative weeks
    def drop_negative_weeks(timeseries, dates, weeks):
        # Find the first positive index
        first_positive_index = next(
            (i for i, week in enumerate(weeks) if week >= 0), None
        )
        if first_positive_index is None:
            return [], [], []
        # Drop all elements before the first positive index
        return (
            timeseries[first_positive_index:],
            dates[first_positive_index:],
            weeks[first_positive_index:],
        )

    # Iterate over each row in the DataFrame
    for index, row in timeseries_df.iterrows():
        # Drop all negative weeks
        (
            timeseries_df.at[index, "Timeseries"],
            timeseries_df.at[index, "Dates"],
            timeseries_df.at[index, "Weeks"],
        ) = drop_negative_weeks(row["Timeseries"], row["Dates"], row["Weeks"])

    return timeseries_df


def gap_fill_dates(timeseries_df, date_col="Dates", ts_cols=["Timeseries"]):

    def get_all_dates(timeseries_df, date_col):
        all_dates = []
        for _, row in timeseries_df.iterrows():
            # row['num_dates'] = len(row[date_col])
            # check if any items in the list are not in all_dates
            for date in row[date_col]:
                if date not in all_dates:
                    all_dates.append(date)

        return all_dates

    all_dates = get_all_dates(timeseries_df, date_col)

    # Function to reindex the timeseries and Date columns
    def reindex_timeseries_and_dates(timeseries, dates, all_dates):
        # Create a DataFrame from the timeseries and Date columns
        df = pd.DataFrame({"Date": dates, "Value": timeseries})
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Reindex the DataFrame using all_dates
        all_dates_index = pd.to_datetime(all_dates)
        reindexed_df = df.reindex(all_dates_index, fill_value=np.nan)
        # sort according to date
        reindexed_df = reindexed_df.sort_index()

        # Convert the reindexed DataFrame back to lists
        reindexed_dates = reindexed_df.index.tolist()
        reindexed_timeseries = reindexed_df["Value"].tolist()

        # print(reindexed_dates, reindexed_timeseries)
        return reindexed_dates, reindexed_timeseries

    # Iterate over each row in the DataFrame
    for index, row in timeseries_df.iterrows():
        # Reindex the timeseries and Date columns
        for ts_col in ts_cols:
            reindexed_dates, reindexed_timeseries = reindex_timeseries_and_dates(
                row[ts_col], row[date_col], all_dates
            )
            # Save the reindexed lists back to the DataFrame
            timeseries_df.at[index, date_col] = reindexed_dates
            timeseries_df.at[index, ts_col] = reindexed_timeseries
    for ts_col in ts_cols:
        timeseries_df[ts_col] = timeseries_df[ts_col].apply(
            lambda ts: pd.Series(ts).interpolate(limit_direction="both").values
        )

    return timeseries_df


def str2date(date):
    # check if it is of type datetime
    if isinstance(date, datetime):
        return date
    return datetime.strptime(date, "%Y-%m-%d")


def convert_tuple2dates(date_range):
    if isinstance(date_range[0], str):
        out1 = str2date(date_range[0])
    else:
        out1 = date_range[0]
    if isinstance(date_range[1], str):
        out2 = str2date(date_range[1])
    else:
        out2 = date_range[1]
    return out1, out2


def stringify_date(date):
    # check if its a list
    if not isinstance(date, list):
        date = [date]

    return [d.strftime("%Y-%m-%d") for d in date]


def count_days(ref_date, date):
    ref_date, date = convert_tuple2dates([ref_date, date])
    return (ref_date - date).days


def get_planting_days(row, start_date):
    planting_date = row["Planting_date"]
    return count_days(planting_date, start_date)


def find_min_max_mean_stdev(timeseries):
    if len(timeseries) == 0:
        return 0, 0, 0, 0
    try:
        out = (
            np.min(timeseries),
            np.max(timeseries),
            np.mean(timeseries),
            np.std(timeseries),
        )
    except:
        out = 0, 0, 0, 0

    return out


def generate_column_names(variables, k, era5_variables=None):
    base_columns = ["troughs_values", "crests_values", "trough_days", "crest_days"]
    columns = []
    for var in variables:
        for base in base_columns:
            for i in range(k):
                columns.append(f"{var}_{base}_{i+1}")
        columns.extend([f"{var}_num_troughs", f"{var}_num_crests"])

        if era5_variables:
            for era5_var in era5_variables:
                for event in ["crest", "trough"]:
                    for i in range(k):
                        columns.append(f"{var}_{event}_{era5_var}_{i+1}")

    return columns


def count_inflection_points(timeseries):
    if len(timeseries) < 3:
        return 0  # Not enough points to have an inflection point
    second_derivative = np.diff(np.diff(timeseries))
    sign_changes = np.diff(np.sign(second_derivative))
    return np.sum(sign_changes != 0)


def find_troughs_and_crests(
    row, start_date, date_column, variables, k=1, era5_variables=None
):
    start_date = pd.to_datetime(start_date)
    dates = row[date_column]

    def create_empty_series():
        empty_data = {}
        for var in variables:
            for i in range(k):
                empty_data.update(
                    {
                        f"{var}_troughs_values_{i+1}": 0,
                        f"{var}_crests_values_{i+1}": 0,
                        f"{var}_trough_days_{i+1}": 0,
                        f"{var}_crest_days_{i+1}": 0,
                    }
                )
            empty_data.update(
                {
                    f"{var}_num_troughs": 0,
                    f"{var}_num_crests": 0,
                }
            )
            if era5_variables:
                for era5_var in era5_variables:
                    for i in range(k):
                        empty_data.update(
                            {
                                f"{var}_crest_{era5_var}_{i+1}": np.nan,
                                f"{var}_trough_{era5_var}_{i+1}": np.nan,
                            }
                        )
        return pd.Series(empty_data)

    def get_season_average(ts):
        return np.mean(ts)

    def get_trough_crest_dates(ts, dates, k):

        # check if ts is empty
        if len(ts) == 0:
            return None, None, [], [], 0, 0

        ts = np.array(ts)

        try:
            crests_indices, _ = find_peaks(ts)
            troughs_indices, _ = find_peaks(-ts)

            num_crests = len(crests_indices)
            num_troughs = len(troughs_indices)

            crests_indices = crests_indices[:k]
            troughs_indices = troughs_indices[:k]

            crests_indices = crests_indices[crests_indices < len(dates)]
            troughs_indices = troughs_indices[troughs_indices < len(dates)]

            crests_dates = (
                [dates[i] for i in crests_indices] if len(crests_indices) > 0 else []
            )
            troughs_dates = (
                [dates[i] for i in troughs_indices] if len(troughs_indices) > 0 else []
            )
        except:
            return None, None, [], [], 0, 0

        return (
            crests_dates,
            troughs_dates,
            crests_indices,
            troughs_indices,
            num_crests,
            num_troughs,
        )

    def count_days(date, start_date):
        return (pd.to_datetime(date) - start_date).days

    results = {}

    # check if dates is not an np.array
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    try:
        if len(dates) <= 1:
            print("Warning: Dates too small, returning empty troughs and crests", dates)
            return create_empty_series()
    except:
        print("Warning: Unsized object, returning empty troughs and crests", dates)
        return create_empty_series()

    max_days = count_days(dates[-1], start_date)

    for var in variables:

        try:
            ts = row[var]
            season_avg = get_season_average(ts)

            (
                crests_dates,
                troughs_dates,
                crests_indices,
                troughs_indices,
                num_crests,
                num_troughs,
            ) = get_trough_crest_dates(ts, dates, k)

            crest_days = (
                [count_days(date, start_date) for date in crests_dates]
                if crests_dates
                else []
            )
            trough_days = (
                [count_days(date, start_date) for date in troughs_dates]
                if troughs_dates
                else []
            )

            crests_values = (
                [ts[i] - season_avg for i in crests_indices]
                if len(crests_indices) > 0
                else []
            )
            troughs_values = (
                [ts[i] - season_avg for i in troughs_indices]
                if len(troughs_indices) > 0
                else []
            )

            # Pad lists to ensure they have length k
            crests_values += [0] * (k - len(crests_values))
            troughs_values += [0] * (k - len(troughs_values))
            crest_days += [max_days] * (k - len(crest_days))  # replace by max days
            trough_days += [max_days] * (k - len(trough_days))  # replace by max days

            for i in range(k):
                results.update(
                    {
                        f"{var}_troughs_values_{i+1}": troughs_values[i],
                        f"{var}_crests_values_{i+1}": crests_values[i],
                        f"{var}_trough_days_{i+1}": trough_days[i],
                        f"{var}_crest_days_{i+1}": crest_days[i],
                    }
                )

            results.update(
                {
                    f"{var}_num_troughs": num_troughs,
                    f"{var}_num_crests": num_crests,
                }
            )

            if era5_variables:
                for era5_var in era5_variables:
                    crest_era5 = (
                        [row[f"ts_daily_{era5_var}"][i] for i in crests_indices]
                        if len(crests_indices) > 0
                        else []
                    )
                    trough_era5 = (
                        [row[f"ts_daily_{era5_var}"][i] for i in troughs_indices]
                        if len(troughs_indices) > 0
                        else []
                    )

                    # Pad ERA5 lists to ensure they have length k
                    crest_era5 += [np.nan] * (k - len(crest_era5))
                    trough_era5 += [np.nan] * (k - len(trough_era5))

                    for i in range(k):
                        results.update(
                            {
                                f"{var}_crest_{era5_var}_{i+1}": crest_era5[i],
                                f"{var}_trough_{era5_var}_{i+1}": trough_era5[i],
                            }
                        )
        except:
            return create_empty_series()

    return pd.Series(results)


import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def generate_s1_combined_spline(
    timeseries, start_date, end_date, mode="normal", sigma=None
):
    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate x_smooth for the entire date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    x_smooth = (date_range - start_date).days.values

    # Create a list to store the spline results for each row
    spline_results = []

    for _, row in timeseries.iterrows():
        row_result = {}

        # Combine Ascending and Descending data
        dates = []
        vv_values = []
        vh_values = []

        for orbit in ["ASCENDING", "DESCENDING"]:
            if f"dates_{orbit}" in row.index:
                dates.extend(pd.to_datetime(row[f"dates_{orbit}"]))
                vv_values.extend(row[f"ts_VV_{orbit}"])
                vh_values.extend(row[f"ts_VH_{orbit}"])

        # Sort the combined data by date
        combined_data = sorted(zip(dates, vv_values, vh_values))
        dates, vv_values, vh_values = zip(*combined_data)

        # Convert dates to numerical values (days since the start_date)
        x = [(d - start_date).days for d in dates]

        if mode == "temporal_smoothing" and sigma is not None:
            # Apply Gaussian filter for temporal smoothing
            vv_values = gaussian_filter1d(vv_values, sigma)
            vh_values = gaussian_filter1d(vh_values, sigma)

        # Fit cubic spline to VV and VH time series
        cs_vv = CubicSpline(x, vv_values)
        cs_vh = CubicSpline(x, vh_values)

        # Calculate smooth y values over the entire date range
        y_smooth_vv = cs_vv(x_smooth)
        y_smooth_vh = cs_vh(x_smooth)

        # Store the results
        row_result["ts_VV_combined_spline"] = y_smooth_vv
        row_result["ts_VH_combined_spline"] = y_smooth_vh
        row_result["dates_spline"] = date_range.strftime("%Y-%m-%d").tolist()
        # row_result['dates_spline'] = date_range

        # ratios
        row_result["ts_spline_ratio"] = y_smooth_vv - y_smooth_vh

        # Store original combined data
        row_result["dates_combined"] = dates
        row_result["ts_VV_combined"] = vv_values
        row_result["ts_VH_combined"] = vh_values

        spline_results.append(row_result)

    # Create a new dataframe with the spline results
    spline_df = pd.DataFrame(spline_results, index=timeseries.index)

    # Combine the original timeseries with the spline results
    result_df = pd.concat([timeseries, spline_df], axis=1)

    return result_df


def generate_s1_spline(
    timeseries,
    start_date,
    end_date,
    mode="normal",
    sigma=None,
    prefix="ts_",
    variable=None,
    convert_to_db=False,
    do_gaussian=True,
    ratio_smoothing_sigma=10,
    orbits=["ASCENDING", "DESCENDING"],
):
    """
    Inputs:
    timeseries: pd.DataFrame
    start_date: str
    end_date: str
    mode: str (normal, temporal_smoothing)
    sigma: float -> standard deviation for Gaussian filter for temporal smoothing
    prefix: str -> prefix for the columns in the timeseries
    variable: str -> variable to use for spline generation
    do_gaussian: bool -> whether to fit a Gaussian curve to the ratios
    ratio_smoothing_sigma: float -> standard deviation for Gaussian filter for ratio smoothing


    """

    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    prefix = "" if prefix == None else prefix

    # Generate x_smooth for the entire date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    x_smooth = (date_range - pd.to_datetime(start_date)).days.values

    # Identify the orbit types
    if f"{prefix}VV" in timeseries.columns:
        orbits = [""]  # No specific orbit, just VV and VH

    # Create a list to store the spline results for each row
    spline_results = []

    def func(x):
        if convert_to_db:
            return linear_to_db(x)
        return x

    for _, row in timeseries.iterrows():
        row_result = {}
        for orbit in orbits:

            orbit_suffix = f"{orbit}" if orbit else ""
            variable_suffix = f"_{variable}" if variable else ""
            vv_col = f"{prefix}VV_{orbit_suffix}{variable_suffix}"
            vh_col = f"{prefix}VH_{orbit_suffix}{variable_suffix}"
            dates_col = f"dates_{orbit_suffix}"

            # print(f'Processing orbit: {orbit}: {vv_col}, {vh_col}, {dates_col}')

            if vv_col not in row.index or vh_col not in row.index:
                continue

            # Extract the time series and dates for this row
            dates = pd.to_datetime(row[dates_col])
            vv_values = np.array(row[vv_col])
            vh_values = np.array(row[vh_col])

            # Convert dates to numerical values (days since the start)
            x = (dates - start_date).days.values

            if mode == "temporal_smoothing" and sigma is not None:
                # Apply Gaussian filter for temporal smoothing
                vv_values = gaussian_filter1d(vv_values, sigma)
                vh_values = gaussian_filter1d(vh_values, sigma)
            # Fit cubic spline to VV and VH time series
            # check if vv_values contains nans
            try:
                cs_vv = CubicSpline(x, vv_values)
                y_smooth_vv = cs_vv(x_smooth)
            except Exception as e:
                y_smooth_vv = [0] * len(x_smooth)
                print(f"Error: {vv_col} spline could not be generated. Dropping row")
                print(f"Error message: {str(e)}")
                print(f"Dates: {x}, vv_values: {vv_values}")
                continue

            try:
                cs_vh = CubicSpline(x, vh_values)
                y_smooth_vh = cs_vh(x_smooth)
            except Exception as e:
                y_smooth_vh = [0] * len(x_smooth)
                print(f"Error: {vh_col} spline could not be generated. Dropping row")
                print(f"Error message: {str(e)}")
                print(f"Dates: {x}, vh_values: {vh_values}")
                continue

            # Store the results
            row_result[f"{vv_col}_spline"] = func(y_smooth_vv)
            row_result[f"{vh_col}_spline"] = func(y_smooth_vh)

            # print(f'Saving spline results for {vv_col}_spline, {vh_col}_spline')

            row_result["dates_spline"] = date_range.strftime("%Y-%m-%d").tolist()

            # ratios
            if convert_to_db:
                row_result[f"{prefix}{orbit_suffix}_spline_ratio"] = (
                    row_result[f"{vv_col}_spline"] - row_result[f"{vh_col}_spline"]
                )
            else:
                row_result[f"{prefix}{orbit_suffix}_spline_ratio"] = [
                    vv / (vh + 1e-9)
                    for vv, vh in zip(
                        row_result[f"{vv_col}_spline"], row_result[f"{vh_col}_spline"]
                    )
                ]
                # row_result[f'{prefix}{orbit_suffix}_spline_ratio'] = row_result[f'{vv_col}_spline'] / row_result[f'{vh_col}_spline']

            if do_gaussian:
                # convert ratios to np.array
                ratios = db_to_linear(
                    np.array(row_result[f"{prefix}{orbit_suffix}_spline_ratio"])
                )
                ratios_normed_lin = _normalize_ratios(ratios)
                if (
                    np.isnan(ratios_normed_lin).any()
                    or np.isinf(ratios_normed_lin).any()
                ):
                    # print(f'Error: {vv_col} spline ratios contain nans or inf. skipping')
                    continue

                # Smooth ratios
                ratios_normed_smooth = gaussian_filter1d(
                    ratios_normed_lin, ratio_smoothing_sigma
                )

                # Convert dates to numeric values for curve fitting
                date_nums = mdates.date2num(row_result["dates_spline"])

                # check if date_nums contains nans or inf
                if np.isnan(date_nums).any() or np.isinf(date_nums).any():
                    print(f"Error: {vv_col} spline dates contain nans or inf, skipping")
                    continue

                # Fit Gaussian curve
                try:
                    popt, _ = curve_fit(
                        _gaussian,
                        date_nums,
                        ratios_normed_smooth,
                        p0=[1, np.mean(date_nums), np.std(date_nums)],
                    )
                    # Generate points for smooth curve using original dates
                    smooth_curve = _gaussian(date_nums, *popt)
                    r2 = r_squared(ratios_normed_smooth, smooth_curve)
                    # Fit a normal distribution to the data:
                    b = popt[1] - date_nums[0]
                    a, c = popt[0], popt[2]
                except RuntimeError:
                    # print("Error: Could not fit Gaussian curve. Using initial guesses for parameters.")
                    a, b, c, r2 = 0, 0, 0, 0

                (
                    row_result[f"{prefix}{orbit_suffix}_ratio_gaussian_a"],
                    row_result[f"{prefix}{orbit_suffix}_ratio_gaussian_b"],
                    row_result[f"{prefix}{orbit_suffix}_ratio_gaussian_c"],
                    row_result[f"{prefix}{orbit_suffix}_ratio_gaussian_r2"],
                ) = (a, b, c, r2)

        spline_results.append(row_result)

    # Create a new dataframe with the spline results
    spline_df = pd.DataFrame(spline_results, index=timeseries.index)

    # Combine the original timeseries with the spline results
    result_df = pd.concat([timeseries, spline_df], axis=1)

    return result_df


def remove_duplicates(x, band_data):
    """
    Remove duplicate values from x and the corresponding values from each band in band_data.

    Parameters:
    x (np.ndarray): Array of x values.
    band_data (dict): Dictionary where keys are band names and values are arrays of band data.

    Returns:
    np.ndarray: Array of x values with duplicates removed.
    dict: Dictionary of band data with duplicates removed.
    """
    # Identify the unique values and their indices
    unique_x, unique_indices = np.unique(x, return_index=True)

    # Remove duplicates from x
    x_cleaned = unique_x

    # Remove corresponding values from each band
    band_data_cleaned = {band: data[unique_indices] for band, data in band_data.items()}

    return x_cleaned, band_data_cleaned


def cloudmask_interpolate(x, y, x_smooth):
    interpolated = []
    for xi in x_smooth:
        if xi in x:
            interpolated.append(y[x.tolist().index(xi)])
        else:
            prev_idx = max([i for i in range(len(x)) if x[i] < xi], default=None)
            next_idx = min([i for i in range(len(x)) if x[i] > xi], default=None)
            if prev_idx is not None and next_idx is not None:
                interpolated.append(min(y[prev_idx], y[next_idx]))
            elif prev_idx is not None:
                interpolated.append(y[prev_idx])
            elif next_idx is not None:
                interpolated.append(y[next_idx])
            else:
                interpolated.append(np.nan)
    return np.array(interpolated)


def generate_s2_spline(
    timeseries,
    start_date,
    end_date,
    band_names,
    variable,
    mode="normal",
    sigma=None,
    prefix="ts_",
):
    """
    Inputs:
    timeseries: pd.DataFrame
    start_date: str
    end_date: str
    band_names: list of str -> list of band names to generate splines for
    variable: str -> variable to use for spline generation
    mode: str (normal, temporal_smoothing)
    prefix: str -> prefix for the columns in the timeseries
    """

    # Convert start_date and end_date to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    prefix = "" if prefix is None else prefix

    # Generate x_smooth for the entire date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    x_smooth = (date_range - pd.to_datetime(start_date)).days.values

    # Create a list to store the spline results for each row
    spline_results = []

    for idx, row in timeseries.iterrows():
        row_result = {}
        # get the s2 dates
        s2_dates = pd.to_datetime(row["dates_s2"])
        x = (s2_dates - start_date).days.values  # same for all bands
        # print(f'Dates for this row: {s2_dates}, {x}')

        # Prepare band data
        band_data = {}
        for band_name in band_names:
            band_name_col = f"{prefix}{band_name}_{variable}"
            if band_name_col in timeseries.columns:
                band_data[band_name_col] = np.array(row[band_name_col])
            if band_name == "cloudmask":
                band_data["cloudmask"] = np.array(row["cloudmask"])
            # else:
            #     band_data[band_name_col] = np.array(row[band_name_col])

        # clean up any duplcate dates
        # TODO: this seems to happen randomly for S2 data
        # Remove duplicates
        x, band_data = remove_duplicates(x, band_data)

        for band_name_col, y in band_data.items():
            print(f"[{idx}]{band_name_col}: {len(x)}, {len(y)}")
            spline_name = f"{band_name_col}_spline"

            # for any missing values
            y = pd.Series(y).interpolate(method="linear").to_numpy()

            if band_name_col == "cloudmask":
                # use nearest neighbour interpolation for cloudmask
                row_result[spline_name] = cloudmask_interpolate(x, y, x_smooth)

            else:
                if mode == "temporal_smoothing" and sigma is not None:
                    y = gaussian_filter1d(y, sigma=sigma)
                spline = CubicSpline(x, y)
                row_result[spline_name] = spline(x_smooth)

        spline_results.append(row_result)

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(spline_results, index=timeseries.index)

    result_df = pd.concat([timeseries, result_df], axis=1)

    return result_df


def db_to_linear(ts):
    """
    Convert a time series from dB to linear scale.

    Parameters:
    ts (pd.Series or np.ndarray): Time series in dB.

    Returns:
    pd.Series or np.ndarray: Time series in linear scale.
    """
    return 10 ** (ts / 10)


def linear_to_db(ts):
    """
    Convert a time series from linear scale to dB.

    Parameters:
    ts (pd.Series or np.ndarray): Time series in linear scale.

    Returns:
    pd.Series or np.ndarray: Time series in dB.
    """
    ts = np.maximum(ts, 1e-10)  # Avoid log(0)
    return 10 * np.log10(ts)


def _normalize_ratios(ts_ratio, ignore_sections=[1, 1]):
    if len(ts_ratio) <= ignore_sections[0] + ignore_sections[1]:
        raise ValueError(
            "The time series must have more than 10 entries to ignore the first and last 5 entries."
        )

    # Ignore the first and last 5 entries
    ts_ratio_middle = ts_ratio[ignore_sections[0] : -ignore_sections[1]]

    # Perform normalization on the middle part
    min_val = np.min(ts_ratio_middle)
    max_val = np.max(ts_ratio_middle)

    normalized_middle = [
        (y - min_val) / (max_val - min_val + 1e-6) for y in ts_ratio_middle
    ]

    # Combine the ignored parts with the normalized middle part
    normalized_ts_ratio = np.concatenate(
        (
            [normalized_middle[0]] * ignore_sections[0],
            normalized_middle,
            [normalized_middle[-1]] * ignore_sections[1],
        )
    )
    return normalized_ts_ratio


from scipy.stats import norm


def fit_gaussian(ts):
    """
    Fit a Gaussian distribution to a normalized time series.

    Parameters:
    ts (np.ndarray): Normalized time series.
    sigma (float): Standard deviation for Gaussian smoothing (optional).

    Returns:
    tuple: Mean and standard deviation of the fitted Gaussian distribution.
    """
    # Fit Gaussian distribution to the time series
    mu, std = norm.fit(ts)

    return mu, std


def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-6))


def _gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def plot_s1_splines(
    df,
    ylim=(-30, 0),
    vlines=[],
    hlines=[-15.0, -25.0],
    mode="average",
    plot_ratios=True,
    ratio_smoothing_sigma=10,
    plot_orbits=["ASCENDING"],
    plot_RVI=True,
    title=None,
    ret_avg=True,
    prefix="ts",
    variable=None,
    convert_to_db=False,
):

    prefix = f"{prefix}_" if prefix else ""
    variable = f"_{variable}" if variable else ""
    # Determine which type of data we're dealing with
    if "ts_VV_combined_spline" in df.columns:
        plot_type = "combined"
        orbits = ["combined"]
    else:
        plot_type = "separate"
        orbits = plot_orbits
        if not orbits and f"{prefix}VV" in df.columns:
            orbits = [""]  # No specific orbit

    # Determine the number of columns based on plot_ratios and plot_RVI
    num_cols = 2 if plot_ratios or plot_RVI else 1

    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(7 * num_cols, 7))

    # Ensure axes is always a list
    if num_cols == 1:
        axes = [axes]

    # Calculate opacity based on number of rows
    num_rows = len(df)
    scatter_opacity = max(0.1, min(1, 5 / num_rows))
    line_opacity = max(0.05, min(0.5, 2 / num_rows))
    ret_df = None

    for orbit in orbits:
        orbit_suffix = f"_{orbit}" if orbit else ""

        ax = axes[0]  # Main plot always in first column

        if plot_type == "separate":
            dates_col = f"dates{orbit_suffix}"
            vv_col = f"{prefix}VV{orbit_suffix}{variable}"
            vh_col = f"{prefix}VH{orbit_suffix}{variable}"
            vv_spline_col = f"{prefix}VV{orbit_suffix}{variable}_spline"
            vh_spline_col = f"{prefix}VH{orbit_suffix}{variable}_spline"

            def func(x):
                if convert_to_db:
                    return linear_to_db(x)
                return x

            def db_error_bars(value, std_dev):
                # Calculate upper and lower bounds in linear scale
                upper_linear = value + std_dev
                lower_linear = np.maximum(
                    value - std_dev, 1e-10
                )  # Prevent negative values

                # Convert to dB
                value_db = linear_to_db(value)
                upper_db = linear_to_db(upper_linear)
                lower_db = linear_to_db(lower_linear)

                # Calculate asymmetric error bars
                err_plus = upper_db - value_db
                err_minus = value_db - lower_db

                return err_minus, err_plus

            if mode == "average":
                for _, row in df.iterrows():
                    spline_dates = pd.to_datetime(row["dates_spline"])
                    original_dates = pd.to_datetime(row[dates_col])

                    # Plot original data
                    ax.scatter(
                        original_dates,
                        func(row[vv_col]),
                        color="tab:blue",
                        alpha=scatter_opacity,
                        s=20,
                    )
                    ax.scatter(
                        original_dates,
                        func(row[vh_col]),
                        color="tab:orange",
                        alpha=scatter_opacity,
                        s=20,
                    )

                    # Plot splines
                    ax.plot(
                        spline_dates,
                        func(row[vv_spline_col]),
                        color="tab:blue",
                        alpha=line_opacity,
                    )
                    ax.plot(
                        spline_dates,
                        func(row[vh_spline_col]),
                        color="tab:orange",
                        alpha=line_opacity,
                    )

                # Collect all VV and VH splines
                vv_splines = np.array(
                    [db_to_linear(row[vv_spline_col]) for _, row in df.iterrows()]
                )
                vh_splines = np.array(
                    [db_to_linear(row[vh_spline_col]) for _, row in df.iterrows()]
                )

                avg_vv = np.mean(vv_splines, axis=0)
                avg_vh = np.mean(vh_splines, axis=0)

                # Calculate standard deviation at each point
                std_vv = np.std(vv_splines, axis=0)
                std_vh = np.std(vh_splines, axis=0)

                # Calculate lower and upper bounds in dB
                lower_bound_db_vv, upper_bound_db_vv = linear_to_db(
                    avg_vv - std_vv
                ), linear_to_db(avg_vv + std_vv)
                lower_bound_db_vh, upper_bound_db_vh = linear_to_db(
                    avg_vh - std_vh
                ), linear_to_db(avg_vh + std_vh)
                # Calculate std dev in dB as half the range
                std_dev_db_vv = (upper_bound_db_vv - lower_bound_db_vv) / 2.0
                std_dev_db_vh = (upper_bound_db_vh - lower_bound_db_vh) / 2.0

                # Back to dB
                avg_vv, avg_vh = linear_to_db(avg_vv), linear_to_db(avg_vh)

                spline_dates = pd.to_datetime(df.iloc[0]["dates_spline"])

                # Plot average lines
                ax.plot(spline_dates, avg_vv, color="blue", label="VV Average")
                ax.plot(spline_dates, avg_vh, color="orange", label="VH Average")
                # Plot standard deviation
                ax.fill_between(
                    spline_dates,
                    avg_vv + std_dev_db_vv,
                    avg_vv - std_dev_db_vv,
                    color="tab:blue",
                    alpha=0.2,
                )
                ax.fill_between(
                    spline_dates,
                    avg_vh + std_dev_db_vh,
                    avg_vh - std_dev_db_vh,
                    color="tab:orange",
                    alpha=0.2,
                )
                ax.set_ylim(ylim)

                if plot_ratios or plot_RVI:
                    ax_right = axes[1]
                    popt = None
                    if plot_ratios:
                        ratios = np.array(
                            [
                                db_to_linear(row[f"ts{orbit_suffix}_spline_ratio"])
                                for _, row in df.iterrows()
                            ]
                        )
                        avg_ratio = np.mean(ratios, axis=0)
                        ratios_normed_lin = _normalize_ratios(avg_ratio)
                        # Smooth ratios
                        ratios_normed_smooth = gaussian_filter1d(
                            ratios_normed_lin, ratio_smoothing_sigma
                        )

                        ax_right.plot(
                            spline_dates,
                            ratios_normed_smooth,
                            color="tab:green",
                            label="VV-VH Ratio Average",
                        )
                        ax_right.set_ylabel("Normalized Ratio")

                        # Convert dates to numeric values for curve fitting
                        date_nums = mdates.date2num(spline_dates)

                        # Fit Gaussian curve
                        try:
                            popt, _ = curve_fit(
                                _gaussian,
                                date_nums,
                                ratios_normed_smooth,
                                p0=[1, np.mean(date_nums), np.std(date_nums)],
                            )
                            # Generate points for smooth curve using original dates
                            smooth_curve = _gaussian(date_nums, *popt)
                            r2 = r_squared(ratios_normed_smooth, smooth_curve)
                            ax_right.plot(
                                spline_dates,
                                smooth_curve,
                                color="tab:purple",
                                label="Gaussian Fit",
                            )
                            # Fit a normal distribution to the data:

                            b = popt[1] - date_nums[0]
                            a, c = popt[0], popt[2]
                            ax_right.set_ylim(0, 1)
                        except RuntimeError:
                            print(
                                "Error: Could not fit Gaussian curve. Using initial guesses for parameters."
                            )
                            a, b, c, r2 = 0, 0, 0, 0

                    if plot_RVI:
                        RVI = 4 * avg_vh / (avg_vv + avg_vh + 1e-6)
                        ax_rvi = ax_right.twinx() if plot_ratios else ax_right
                        ax_rvi.plot(
                            spline_dates,
                            RVI,
                            color="tab:red",
                            label="RVI Average",
                            linestyle="--",
                        )
                        ax_rvi.set_ylabel("RVI")
                        ax_rvi.set_ylim(2, 3)
                        ax_rvi.legend(loc="upper right")

                    ax_right.set_title(
                        f"{orbit} Ratios and RVI | a={a:.2f}, b={b:.0f}, c={c:.0f}, R2={r2:.2f}"
                    )
                    ax_right.set_xlabel("Date")
                    ax_right.legend()
                    ax_right.grid(True, which="both", linestyle="--", linewidth=0.5)
                    ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
                    ax_right.xaxis.set_major_locator(mdates.AutoDateLocator())

                if ret_avg:
                    # save them
                    # convert spline_dates to a list of strings, avg_vv and avg_vh to list
                    spline_dates = spline_dates.strftime("%Y-%m-%d").tolist()
                    avg_vv, avg_vh = (
                        db_to_linear(avg_vv).tolist(),
                        db_to_linear(avg_vh).tolist(),
                    )

                    ret_df = pd.DataFrame(
                        {
                            "dates_spline": [spline_dates],
                            "VV": [avg_vv],
                            "VH": [avg_vh],
                            "ratio": [ratios_normed_smooth],
                            "RVI": [RVI],
                            "a": [a],
                            "b": [b],
                            "c": [c],
                            "R2": [r2],
                        }
                    )
                    # display(ret_df)

            elif mode == "detailed":
                for _, row in df.iterrows():
                    spline_dates = pd.to_datetime(row["dates_spline"])
                    original_dates = pd.to_datetime(row[dates_col])
                    stdev_vv_col, stdev_vh_col = (
                        f"{prefix}VV{orbit_suffix}_std",
                        f"{prefix}VH{orbit_suffix}_std",
                    )

                    # Plot original data
                    ax.scatter(
                        original_dates,
                        func(row[vv_col]),
                        color="tab:blue",
                        alpha=scatter_opacity,
                        s=20,
                    )
                    ax.scatter(
                        original_dates,
                        func(row[vh_col]),
                        color="tab:orange",
                        alpha=scatter_opacity,
                        s=20,
                    )
                    # Calculate error bars for VV and VH
                    vv_err = db_error_bars(row[vv_col], row[stdev_vv_col])
                    vh_err = db_error_bars(row[vh_col], row[stdev_vh_col])

                    # Plot error bars
                    ax.errorbar(
                        original_dates,
                        linear_to_db(row[vv_col]),
                        yerr=vv_err,
                        fmt="none",
                        ecolor="tab:blue",
                        alpha=0.9,
                    )
                    ax.errorbar(
                        original_dates,
                        linear_to_db(row[vh_col]),
                        yerr=vh_err,
                        fmt="none",
                        ecolor="tab:orange",
                        alpha=0.9,
                    )

                    # Plot splines
                    ax.plot(
                        spline_dates,
                        func(row[vv_spline_col]),
                        color="tab:blue",
                        alpha=line_opacity,
                    )
                    ax.plot(
                        spline_dates,
                        func(row[vh_spline_col]),
                        color="tab:orange",
                        alpha=line_opacity,
                    )

                if plot_ratios or plot_RVI:
                    ax_right = axes[1]
                    popt = None
                    if plot_ratios:
                        ratios = np.array(
                            [
                                (row[vv_spline_col] / row[vh_spline_col])
                                for _, row in df.iterrows()
                            ]
                        )
                        avg_ratio = np.mean(ratios, axis=0)
                        ratios_normed_lin = _normalize_ratios(avg_ratio)
                        # Smooth ratios
                        ratios_normed_smooth = gaussian_filter1d(
                            ratios_normed_lin, ratio_smoothing_sigma
                        )

                        ax_right.plot(
                            spline_dates,
                            ratios_normed_smooth,
                            color="tab:green",
                            label="VV-VH Ratio Average",
                        )
                        ax_right.set_ylabel("Normalized Ratio")

                        # Convert dates to numeric values for curve fitting
                        date_nums = mdates.date2num(spline_dates)

                        # Fit Gaussian curve
                        try:
                            popt, _ = curve_fit(
                                _gaussian,
                                date_nums,
                                ratios_normed_smooth,
                                p0=[1, np.mean(date_nums), np.std(date_nums)],
                            )
                            # Generate points for smooth curve using original dates
                            smooth_curve = _gaussian(date_nums, *popt)
                            r2 = r_squared(ratios_normed_smooth, smooth_curve)
                            ax_right.plot(
                                spline_dates,
                                smooth_curve,
                                color="tab:purple",
                                label="Gaussian Fit",
                            )
                            # Fit a normal distribution to the data:

                            b = popt[1] - date_nums[0]
                            a, c = popt[0], popt[2]
                            ax_right.set_ylim(0, 1)
                        except RuntimeError:
                            print(
                                "Error: Could not fit Gaussian curve. Using initial guesses for parameters."
                            )
                            a, b, c, r2 = 0, 0, 0, 0

                    if plot_RVI:
                        RVI = (
                            4
                            * func(row[vh_spline_col])
                            / (
                                func(row[vv_spline_col])
                                + func(row[vh_spline_col])
                                + 1e-6
                            )
                        )
                        ax_rvi = ax_right.twinx() if plot_ratios else ax_right
                        ax_rvi.plot(
                            spline_dates,
                            RVI,
                            color="tab:red",
                            label="RVI Average",
                            linestyle="--",
                        )
                        ax_rvi.set_ylabel("RVI")
                        ax_rvi.set_ylim(2, 3)
                        ax_rvi.legend(loc="lower right")

                    ax_right.set_title(
                        f"{orbit} Ratios and RVI | a={a:.2f}, b={b:.0f}, c={c:.0f}, R2={r2:.2f}"
                    )
                    ax_right.set_xlabel("Date")
                    ax_right.legend()
                    ax_right.grid(True, which="both", linestyle="--", linewidth=0.5)
                    ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
                    ax_right.xaxis.set_major_locator(mdates.AutoDateLocator())

        else:  # plot_type == 'combined'
            vv_spline_col = f"ts_VV{orbit_suffix}_spline"
            vh_spline_col = f"ts_VH{orbit_suffix}_spline"

            for _, row in df.iterrows():

                original_dates_asc = pd.to_datetime(row["dates_ASCENDING"])
                original_dates_desc = pd.to_datetime(row["dates_DESCENDING"])
                spline_dates = pd.to_datetime(row["dates_spline"])

                # Plot original data
                ax.scatter(
                    original_dates_asc,
                    row["ts_VV_ASCENDING"],
                    color="darkblue",
                    alpha=scatter_opacity,
                    s=20,
                )
                ax.scatter(
                    original_dates_desc,
                    row["ts_VV_DESCENDING"],
                    color="lightblue",
                    alpha=scatter_opacity,
                    s=20,
                )
                ax.scatter(
                    original_dates_asc,
                    row["ts_VH_ASCENDING"],
                    color="darkred",
                    alpha=scatter_opacity,
                    s=20,
                )
                ax.scatter(
                    original_dates_desc,
                    row["ts_VH_DESCENDING"],
                    color="salmon",
                    alpha=scatter_opacity,
                    s=20,
                )

                # Plot splines
                ax.plot(
                    spline_dates,
                    row[vv_spline_col],
                    color="tab:blue",
                    alpha=line_opacity,
                )
                ax.plot(
                    spline_dates,
                    row[vh_spline_col],
                    color="tab:orange",
                    alpha=line_opacity,
                )

            # Collect all VV and VH splines
            vv_splines = np.array([row[vv_spline_col] for _, row in df.iterrows()])
            vh_splines = np.array([row[vh_spline_col] for _, row in df.iterrows()])

            # Calculate DTW barycenter average
            avg_vv = np.mean(vv_splines, axis=0)
            avg_vh = np.mean(vh_splines, axis=0)

            # Calculate standard deviation at each point
            std_vv = np.std(vv_splines, axis=0)
            std_vh = np.std(vh_splines, axis=0)
            # Plot average lines
            ax.plot(spline_dates, avg_vv, color="blue", label="VV Average")
            ax.plot(spline_dates, avg_vh, color="tab:red", label="VH Average")

            # # Plot standard deviation
            ax.fill_between(
                spline_dates,
                avg_vv - std_vv,
                avg_vv + std_vv,
                color="tab:blue",
                alpha=0.2,
            )
            ax.fill_between(
                spline_dates,
                avg_vh - std_vh,
                avg_vh + std_vh,
                color="tab:orange",
                alpha=0.2,
            )

        # Set up legend
        if mode == "detailed":
            if plot_type == "separate":
                ax.scatter([], [], color="tab:blue", label="VV Original", alpha=1)
                ax.scatter([], [], color="tab:orange", label="VH Original", alpha=1)
                ax.plot([], [], color="tab:blue", label="VV Spline", alpha=1)
                ax.plot([], [], color="tab:orange", label="VH Spline", alpha=1)
            else:  # plot_type == 'combined'
                ax.scatter([], [], color="darkblue", label="VV Ascending", alpha=1)
                ax.scatter([], [], color="lightblue", label="VV Descending", alpha=1)
                ax.scatter([], [], color="darkred", label="VH Ascending", alpha=1)
                ax.scatter([], [], color="salmon", label="VH Descending", alpha=1)
                ax.plot([], [], color="blue", label="VV Spline", alpha=1)
                ax.plot([], [], color="red", label="VH Spline", alpha=1)
        else:  # mode == 'average'
            ax.plot([], [], color="blue", label="VV Average")
            ax.plot([], [], color="red", label="VH Average")

        ax.set_title(f"{orbit} time series for {title}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Backscatter (dB)")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Format x-axis to display dates properly in day-month format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        ax.set_ylim(ylim)
        if len(vlines) > 0:
            ax.axvline(x=vlines, color="black", linestyle="--", linewidth=1)

        if len(hlines) > 0:
            for h in hlines:
                ax.axhline(y=h, color="black", linewidth=0.5)

    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    plt.tight_layout()
    plt.show()
    if ret_avg:
        return ret_df


def plot_s1_spline(
    df,
    row_index=0,
    ylim=(-40, 0),
    vlines=[],
    hlines=[-15.0, -25.0],
):
    # Determine which orbits are present in the data
    orbits = []
    if "ts_VV_ASCENDING" in df.columns:
        orbits.append("ASCENDING")
    if "ts_VV_DESCENDING" in df.columns:
        orbits.append("DESCENDING")
    if not orbits and "ts_VV" in df.columns:
        orbits.append("")  # No specific orbit

    # Create subplots
    fig, axes = plt.subplots(len(orbits), 1, figsize=(12, 6 * len(orbits)), sharex=True)
    if len(orbits) == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot

    for ax, orbit in zip(axes, orbits):
        orbit_suffix = f"_{orbit}" if orbit else ""

        # Plot original VV data
        original_dates = pd.to_datetime(df.loc[row_index, f"dates{orbit_suffix}"])
        ax.scatter(
            original_dates,
            df.loc[row_index, f"ts_VV{orbit_suffix}"],
            color="blue",
            label="VV Original",
            alpha=0.7,
        )

        # Plot original VH data
        ax.scatter(
            original_dates,
            df.loc[row_index, f"ts_VH{orbit_suffix}"],
            color="red",
            label="VH Original",
            alpha=0.7,
        )

        # Plot VV spline
        spline_dates = pd.to_datetime(df.loc[row_index, "dates_spline"])
        ax.plot(
            spline_dates,
            df.loc[row_index, f"ts_VV{orbit_suffix}_spline"],
            color="blue",
            label="VV Spline",
        )

        # Plot VH spline
        ax.plot(
            spline_dates,
            df.loc[row_index, f"ts_VH{orbit_suffix}_spline"],
            color="red",
            label="VH Spline",
        )

        ax.set_title(f'{orbit if orbit else "S1"} Time Series and Splines')
        ax.set_xlabel("Date")
        ax.set_ylabel("Backscatter (dB)")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        # Set y-axis limits
        ax.set_ylim(ylim)
        if len(vlines) > 0:
            ax.axvline(x=vlines, color="black", linestyle="--", linewidth=1)

        if len(hlines) > 0:
            for h in hlines:
                ax.axhline(y=h, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_s1_groups_spline(
    timeseries_list,
    dates_list,
    filter=["ts_VH", "ts_VV"],
    labels=["Control", "DSR", "AWD"],
    colors=["blue", "orange", "red"],
    ylim=(-40, 0),
    vlines=[],
    hlines=[-15.0, -25.0],
    plot_ratios=False,
    figsize=(7, 10),
):

    # Check if there is only one item in the timeseries list
    if len(timeseries_list) == 1:
        _, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        # Create subplots
        _, axes = plt.subplots(1, len(timeseries_list), figsize=(20, 5), sharey=True)

    # convert dates_list to pandas.datetime
    dates_list = [pd.to_datetime(d) for d in dates_list]

    for i, (timeseries, label, color) in enumerate(
        zip(timeseries_list, labels, colors)
    ):
        stdevs = []

        for j, f in enumerate(filter):
            ts_list = timeseries[f].values.copy()
            if j == 0:
                color = f"tab:{color}"

            # Calculate alpha value and ensure it is not greater than 1
            alpha = min(5 / len(ts_list), 1.0)

            # Plot each time series using a fitted spline
            for dates, ts in zip(dates_list, ts_list):
                # Convert dates to numerical values (days) for spline fitting
                x = (dates - dates.min()).days.values

                # Fit cubic spline to this time series
                cs = CubicSpline(x, ts)

                # Generate new dates with daily increments
                date_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
                x_smooth = (date_range - date_range.min()).days.values

                # Calculate smooth y values
                y_smooth = cs(x_smooth)

                # Plot original data points
                axes[i].scatter(dates, ts, color=color, alpha=alpha, s=10)

                # Plot smoothed spline curve
                axes[i].plot(date_range, y_smooth, color=color, alpha=0.7, linewidth=1)

            # Calculate and plot the average
            avg = dtw_barycenter_averaging(ts_list)
            distances = [dtw(ts, avg) for ts in ts_list]
            std = np.std(distances)
            stdevs.append(std)
            # Fit spline to average
            x_avg = (dates_list[0] - dates_list[0].min()).days.values
            cs_avg = CubicSpline(x_avg, avg)

            # Generate new dates with daily increments for average
            date_range_avg = pd.date_range(
                start=dates_list[0].min(), end=dates_list[0].max(), freq="D"
            )
            x_smooth_avg = (date_range_avg - date_range_avg.min()).days.values
            y_smooth_avg = cs_avg(x_smooth_avg)

            axes[i].plot(
                date_range_avg,
                y_smooth_avg,
                color="black",
                label=f"{label}-{f}",
                linewidth=2,
                linestyle="--",
            )

            axes[i].set_xlabel("Date")
            if i == 0:
                axes[i].set_ylabel(f"{filter} Values")
            axes[i].legend()
            # Format the x-axis to display dates as day-month
            date_form = DateFormatter("%d-%m")
            axes[i].xaxis.set_major_formatter(date_form)

            # Set y-axis limits
            axes[i].set_ylim(ylim)
            if len(vlines) > 0:
                axes[i].axvline(x=vlines[i], color="black", linestyle="--", linewidth=1)

            if len(hlines) > 0:
                for h in hlines:
                    axes[i].axhline(y=h, color="black", linewidth=0.1)

        if plot_ratios and len(filter) == 2:
            ts_list = timeseries["ts_VV"].values - timeseries["ts_VH"].values
            # print(ts_list)

            # Calculate alpha value and ensure it is not greater than 1
            alpha = min(5 / len(ts_list), 1.0)

            # Plot each time series
            for dates, ts in zip(dates_list, ts_list):
                axes[i].plot(dates, ts, color="black", alpha=alpha, linewidth=1)

            # Calculate and plot the average
            avg = dtw_barycenter_averaging(ts_list)
            distances = [dtw(ts, avg) for ts in ts_list]

            axes[i].plot(
                dates_list[0],
                avg,
                color="darkgray",
                label=f"{label}-VH/VV",
                linewidth=2.5,
                linestyle="--",
            )

            axes[i].set_xlabel("Date")
            if i == 0:
                axes[i].set_ylabel(f"{filter} Values")
            axes[i].legend()
            # Format the x-axis to display dates as day-month
            date_form = DateFormatter("%d-%m")
            axes[i].xaxis.set_major_formatter(date_form)

            y_lim = (ylim[0], 20)

            # Set y-axis limits
            axes[i].set_ylim(y_lim)
        std = combine_standard_deviations(stdevs)
        axes[i].set_title(f"{label} (N={len(ts_list)}, STD={std:.2f})")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def sample_columns_with_aggregation(
    df: pd.DataFrame,
    variables: List[str],
    date_col: str,
    frequency: int,
    sum_variables: List[str] = ["total_precipitation_sum"],
    convert_to_db: bool = True,
) -> pd.DataFrame:
    """
    Sample the specified columns of a DataFrame at a given frequency using the date column.
    Some variables can be summed over the period instead of sampled.
    Ensures all output lists have exactly the same length.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables (List[str]): List of all columns to process.
        date_col (str): The name of the date column.
        frequency (int): The frequency for resampling (keep every nth value in lists)
        sum_variables (List[str], optional): List of variables that should be summed over the period
            rather than sampled. These variables should also be in the variables list.
        convert_to_db (bool, optional): Whether to convert radar variables to dB. Defaults to True.

    Returns:
        pd.DataFrame: A new DataFrame with the processed data in lists.
    """

    def resample_list(
        lst: List[float], freq: int, max_samples: int = None
    ) -> List[float]:
        """
        Take every nth value from the list, ensuring the output doesn't exceed max_samples.
        """
        sampled = lst[::freq]
        if max_samples is not None:
            sampled = sampled[:max_samples]
        return sampled

    def sum_by_groups(lst: List[float], freq: int, max_samples: int) -> List[float]:
        """
        Sum values in groups of size freq, ensuring the output matches max_samples.
        """
        arr = np.array(lst)
        result = []

        for i in range(max_samples):
            start_idx = i * freq
            end_idx = min(start_idx + freq, len(arr))

            if start_idx < len(arr):
                group_sum = arr[start_idx:end_idx].sum()
                result.append(group_sum)
            else:
                break

        return result

    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    sum_variables = sum_variables or []

    def calculate_max_samples(row, date_col, variables, frequency):
        """Calculate the maximum number of samples possible for this row"""
        # Check if any of the required columns are missing or empty
        if date_col not in row or not row[date_col] or len(row[date_col]) == 0:
            return 0

        # Start with date column length
        lengths = [len(row[date_col]) // frequency]

        # Add variable lengths, checking if each variable exists and has data
        for var in variables:
            if var in row and row[var] is not None and len(row[var]) > 0:
                lengths.append(len(row[var]) // frequency)
            else:
                return 0  # If any variable is missing or empty, return 0
        # print(f'DEBUG: DATES: {len(row[date_col])} // {frequency}, MIN LENGTHS: {min(lengths)}, MAX LENGTHS: {max(lengths)}')

        # Return single minimum value
        # return min(lengths) if lengths else 0
        # Return single minimum value, ensuring it's at least 1
        min_length = min(lengths) if lengths else 0
        return max(1, min_length)  # Ensure at least 1 sample is returned if possible

    # Calculate max samples for each row
    processed_df["_max_samples"] = processed_df.apply(
        lambda row: calculate_max_samples(row, date_col, variables, frequency), axis=1
    )

    # Drop rows where max_samples is 0
    zero_samples_mask = processed_df["_max_samples"] == 0
    if zero_samples_mask.any():
        print(
            f"Warning: Dropping {zero_samples_mask.sum()} rows with insufficient data for sampling at frequency {frequency}."
        )
        processed_df = processed_df[~zero_samples_mask].copy()

        # If all rows were dropped, return an empty DataFrame
        if len(processed_df) == 0:
            # print("Warning: All rows were dropped due to insufficient data.")
            raise ValueError("All rows were dropped due to insufficient data.")
            return processed_df

    # Process each variable ensuring consistent lengths
    # Start with the date column
    processed_df[date_col] = processed_df.apply(
        lambda row: resample_list(row[date_col], frequency, row["_max_samples"]), axis=1
    )

    # Process other variables
    for var in variables:
        if var in sum_variables:
            processed_df[var] = processed_df.apply(
                lambda row: sum_by_groups(row[var], frequency, row["_max_samples"]),
                axis=1,
            )
        else:
            # Use a list comprehension to avoid returning a DataFrame
            processed_df[var] = [
                resample_list(row[var], frequency, row["_max_samples"])
                for _, row in processed_df.iterrows()
            ]
            # Convert radar variables to dB if needed
            if (
                convert_to_db
                and "ratio" not in var
                and ("DESCENDING" in var or "ASCENDING" in var)
            ):
                processed_df[var] = [
                    [linear_to_db(v) for v in x] for x in processed_df[var]
                ]

    # Validate that all lists have the same length
    def validate_lengths(row):
        lengths = [len(row[date_col])] + [len(row[var]) for var in variables]
        assert len(set(lengths)) == 1, f"Inconsistent lengths found: {lengths}"
        return True

    try:
        processed_df.apply(validate_lengths, axis=1)
    except AssertionError as e:
        raise ValueError(f"Validation failed: {str(e)}")

    # Remove the temporary column used for max_samples
    processed_df = processed_df.drop("_max_samples", axis=1)

    return processed_df


from sklearn.decomposition import PCA


def apply_pca_to_column(df, column_name, n_components):
    """
    Apply PCA to the values in a column that contains lists and reduce it to the first N components.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing lists.
    n_components (int): The number of components to reduce to.

    Returns:
    pd.DataFrame: A new DataFrame with the reduced components.
    """
    # Extract the column with lists
    data = df[column_name].tolist()

    # Convert the list of lists to a 2D array
    data_matrix = pd.DataFrame(data).values

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_matrix)
    pca_names = [f"{column_name}_PC{i+1}" for i in range(n_components)]

    # Create a new DataFrame with the reduced components
    reduced_df = pd.DataFrame(reduced_data, columns=pca_names)

    # Concatenate the reduced components with the original DataFrame
    result_df = pd.concat([df.drop(columns=[column_name]), reduced_df], axis=1)

    return result_df


def flatten_features(df, column_name):
    # Extract the 'presto_features' column
    features = df[column_name].apply(pd.Series)

    # Rename columns to have a prefix 'feature_'
    features.columns = [f"{column_name}_{i}" for i in range(features.shape[1])]

    # Concatenate the new columns back to the original DataFrame
    df = pd.concat([df.drop(columns=[column_name]), features], axis=1)

    return df


def flatten_2d_features(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Get the shape of the 2D array (assuming all rows have the same shape)
    sample_array = df[column_name].iloc[0]
    if not isinstance(sample_array, np.ndarray) or sample_array.ndim != 2:
        raise ValueError(f"Column '{column_name}' must contain 2D numpy arrays.")

    rows, cols = sample_array.shape  # weeks, features
    num_temp_features = int((rows - 1))

    # Create new column names

    time_features = [
        f"time_{i}_{j}" for i in range(num_temp_features) for j in range(cols)
    ]
    # s1_features = [f's1_{i}_{j}' for i in range(num_temp_features) for j in range(cols)]
    # era5_features = [f'era5_{i}_{j}' for i in range(num_temp_features, 2*num_temp_features) for j in range(cols)]
    location_features = [
        f"location_{i}_{j}" for i in range(num_temp_features, rows) for j in range(cols)
    ]

    # Create new column names
    new_columns = time_features + location_features

    # Flatten the 2D arrays and create a new DataFrame
    flattened_data = np.vstack(df[column_name].apply(lambda x: x.flatten()))

    # Create a new DataFrame with flattened columns
    flattened_df = pd.DataFrame(flattened_data, columns=new_columns, index=df.index)

    # Concatenate the original DataFrame (excluding the original column) with the flattened DataFrame
    result_df = pd.concat([df.drop(columns=[column_name]), flattened_df], axis=1)

    return result_df


def apply_pca_to_2d_column(df, column_name, n_components):
    """
    Apply PCA to a 2D variable stored in a DataFrame column.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing 2D data.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: A new DataFrame with the PCA results added as new columns.
    """
    # Extract the 2D data from the DataFrame column
    data_2d = np.array(df[column_name].tolist())

    # Reshape the data if necessary
    # Assuming each element in the column is a 2D array of shape (m, n)
    # We reshape to (-1, n)
    reshaped_data = data_2d.reshape(-1, data_2d.shape[2])

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)
    print(pca.explained_variance_ratio_)

    # Reshape the result back to (num_samples, m, n_components)
    pca_reshaped = pca_result.reshape(data_2d.shape[0], data_2d.shape[1], n_components)

    # Create a new DataFrame with the PCA results
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"{column_name}_PC{i+1}"] = [pc[:, i] for pc in pca_reshaped]

    return result_df


def apply_pca_to_2d_column_row_wise(df, column_name, n_components):
    """
    Apply PCA to each 2D array in a DataFrame column independently.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing 2D data.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: A new DataFrame with the PCA results added as new columns.
    """

    def pca_transform(array_2d):
        pca = PCA(n_components=n_components)
        y = pca.fit_transform(array_2d)
        # print(pca.explained_variance_ratio_)
        return y

    # Apply PCA to each 2D array in the column
    pca_results = df[column_name].apply(pca_transform)

    # Create a new DataFrame with the PCA results
    result_df = df.copy()
    for i in range(n_components):
        result_df[f"{column_name}_PC{i+1}"] = pca_results.apply(lambda x: x[:, i])

    return result_df


class RandomBaseline:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.y = y

    def predict(self, X):
        return np.random.choice(self.y, size=X.shape[0])
