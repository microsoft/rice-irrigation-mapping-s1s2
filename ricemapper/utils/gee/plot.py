# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw
from matplotlib.dates import DateFormatter
import math


def combine_standard_deviations(stdevs):
    """
    Combine standard deviations from equal-sized datasets.

    Args:
    stdevs (list of float): List of standard deviations to combine.

    Returns:
    float: The combined standard deviation.

    """
    if not stdevs:
        raise ValueError("The list of standard deviations cannot be empty.")

    # Sum of squared standard deviations
    sum_squared = sum(sd**2 for sd in stdevs)

    # Calculate the mean of squared standard deviations
    mean_squared = sum_squared / len(stdevs)

    # Take the square root to get the combined standard deviation
    combined_stdev = math.sqrt(mean_squared)

    return combined_stdev


from scipy.ndimage import gaussian_filter1d


def temporal_gaussian_filter(time_series, sigma):
    """
    Apply a temporal Gaussian filter to smooth a time series.

    Parameters:
    time_series (array-like): Input time series data
    sigma (float): Standard deviation for Gaussian kernel

    Returns:
    array: Smoothed time series
    """
    return gaussian_filter1d(time_series, sigma=sigma)


def plot_s1_groups(
    timeseries_list,
    dates_list,
    filter=["ts_VH", "ts_VV"],
    labels=["Control", "DSR", "AWD"],
    colors=["blue", "orange", "red"],
    ylim=(-40, 0),
    vlines=[],
    hlines=[-15.0, -25.0],
    plot_ratios=False,
    sigma=None,
    figsize=(7, 7),
):

    # Check if there is only one item in the timeseries list
    if len(timeseries_list) == 1:
        _, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        # Create subplots
        _, axes = plt.subplots(1, len(timeseries_list), figsize=(20, 5), sharey=True)

    for i, (timeseries, label, color) in enumerate(
        zip(timeseries_list, labels, colors)
    ):
        stdevs = []

        for j, f in enumerate(filter):
            ts_list = timeseries[f].values.copy()
            if j > 0:
                color = f"tab:{color}"

            # Calculate alpha value and ensure it is not greater than 1
            alpha = min(25 / len(ts_list), 1.0)

            if sigma is not None:
                ts_list = [temporal_gaussian_filter(ts, sigma) for ts in ts_list]

            # Plot each time series
            for dates, ts in zip(dates_list, ts_list):
                axes[i].plot(dates, ts, color=color, alpha=alpha, linewidth=1)

            # Calculate and plot the average
            avg = dtw_barycenter_averaging(ts_list)
            distances = [dtw(ts, avg) for ts in ts_list]
            std = np.std(distances)
            stdevs.append(std)
            axes[i].plot(
                dates_list[0],
                avg,
                color="darkgray",
                label=label,
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

            # Set y-axis limits
            axes[i].set_ylim(ylim)
            if len(vlines) > 0:
                axes[i].axvline(x=vlines[i], color="black", linestyle="--", linewidth=1)

            if len(hlines) > 0:
                for h in hlines:
                    axes[i].axhline(y=h, color="black", linewidth=0.1)

        if plot_ratios and len(filter) == 2:
            ts_list = timeseries["ts_VH"].values / (timeseries["ts_VV"].values + 1e-6)
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
                label=label,
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

            y_lim = (ylim[0], 5)

            # Set y-axis limits
            axes[i].set_ylim(y_lim)
        std = combine_standard_deviations(stdevs)
        axes[i].set_title(f"{label} (N={len(ts_list)}, STD={std:.2f})")

    # Adjust layout
    plt.tight_layout()
    plt.show()


from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from datetime import timedelta
import pandas as pd


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


def plot_s1_group_ratios(
    timeseries_list,
    dates_list,
    labels=["Control", "DSR", "AWD"],
    colors=["tab:blue", "orange", "tab:red"],
    ylim=(-50, 0),
):

    # Create subplots
    _, axes = plt.subplots(1, len(timeseries_list), figsize=(20, 5), sharey=True)

    for i, (timeseries, label, color) in enumerate(
        zip(timeseries_list, labels, colors)
    ):

        ts_list = timeseries["ts_VH"].values / (timeseries["ts_VV"].values + 1e-6)
        # print(ts_list)

        # Calculate alpha value and ensure it is not greater than 1
        alpha = min(25 / len(ts_list), 1.0)

        # Plot each time series
        for dates, ts in zip(dates_list, ts_list):
            axes[i].plot(dates, ts, color=color, alpha=alpha, linewidth=1)

        # Calculate and plot the average
        avg = dtw_barycenter_averaging(ts_list)
        distances = [dtw(ts, avg) for ts in ts_list]
        std = np.std(distances)
        axes[i].plot(
            dates_list[0],
            avg,
            color="darkgray",
            label=label,
            linewidth=2.5,
            linestyle="--",
        )
        axes[i].set_title(f"{label} (N={len(ts_list)}, STD={std:.2f})")
        axes[i].set_xlabel("Date")
        if i == 0:
            axes[i].set_ylabel(f"{filter} Values")
        axes[i].legend()
        # Format the x-axis to display dates as day-month
        date_form = DateFormatter("%d-%m")
        axes[i].xaxis.set_major_formatter(date_form)

        # Set y-axis limits
        # axes[i].set_ylim(ylim)

    # Adjust layout
    plt.tight_layout()
    plt.show()
