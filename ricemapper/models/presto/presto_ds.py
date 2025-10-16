# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from ricemapper.models.presto.utils import (
    S1_BANDS,
    ERA5_BANDS,
    construct_single_presto_input,
)
import torch


class PrestoDataset(Dataset):
    def __init__(
        self,
        source: Union[str, Path, pd.DataFrame],
        transform=None,
        input_s1_bands=["VV", "VH"],
        input_era5_bands=[
            "ts_monthly_temperature_2m",
            "ts_monthly_total_precipitation_sum",
        ],
        input_s2_bands=[],
        cloudmask_band=None,  # or 'cloudmask'
        do_era5=True,
        date_col="dates_s1",
        lat_col="latitude",
        lon_col="longitude",
        label_col="label",
        date_in_weeks=False,
    ):

        if isinstance(source, str) or isinstance(source, Path):
            self.df = pd.read_parquet(source)

        elif isinstance(source, pd.DataFrame):
            self.df = source

        else:
            raise ValueError("source must be a string path or a pandas DataFrame")

        self.do_era5 = do_era5
        self.transform = transform
        self.input_s1_bands = input_s1_bands
        self.input_era5_bands = input_era5_bands
        self.date_col = date_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.label_col = label_col
        self.input_s2_bands = input_s2_bands
        self.cloudmask_band = cloudmask_band
        self.date_in_weeks = date_in_weeks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # print(f'Processing row {idx}')

        s1_data = row[self.input_s1_bands].values
        s1_data = torch.from_numpy(
            np.stack(s1_data, axis=0).T
        ).float()  # (time, bands) -> e.g. 17,2
        dates = pd.to_datetime(row[self.date_col])

        label = torch.tensor(row[self.label_col]).long()

        if not self.do_era5 and len(self.input_s2_bands) > 0:
            # print('S2 data present')
            s2_data = row[self.input_s2_bands].values
            s2_data = torch.from_numpy(np.stack(s2_data, axis=0).T).float()

            if self.cloudmask_band is not None:
                cloudmask = row[self.cloudmask_band]
                # convert to torch boolen
                cloudmask = torch.tensor(cloudmask).float()
            else:
                cloudmask = None

            x, mask, dynamic_world = construct_single_presto_input(
                s1=s1_data,
                s1_bands=S1_BANDS,
            )

        # Check if all required columns are present in the row
        elif all(var in row.index for var in self.input_era5_bands) and self.do_era5:
            era5_data = row[[var for var in self.input_era5_bands]].values
            era5_data = torch.from_numpy(np.stack(era5_data, axis=0).T).float()

            # Check lengths and extend era5_data if necessary
            if era5_data.shape[0] < s1_data.shape[0]:
                last_value = era5_data[-1, :].unsqueeze(0)
                padding = last_value.repeat(s1_data.shape[0] - era5_data.shape[0], 1)
                era5_data = torch.cat((era5_data, padding), dim=0)

            if len(self.input_s2_bands) > 0:
                s2_data = row[self.input_s2_bands].values
                s2_data = torch.from_numpy(np.stack(s2_data, axis=0).T).float()
                s2_mask = row[self.cloudmask_band].values

                x, mask, dynamic_world = construct_single_presto_input(
                    s1=s1_data,
                    s1_bands=S1_BANDS,
                    era5=era5_data,
                    era5_bands=ERA5_BANDS,
                    s2=s2_data,
                    s2_bands=self.input_s2_bands,
                    s2_mask=s2_mask,
                )
            else:
                x, mask, dynamic_world = construct_single_presto_input(
                    s1=s1_data, s1_bands=S1_BANDS, era5=era5_data, era5_bands=ERA5_BANDS
                )

        else:
            x, mask, dynamic_world = construct_single_presto_input(
                s1=s1_data, s1_bands=S1_BANDS
            )

        latlons = torch.tensor([row[self.lat_col], row[self.lon_col]]).float()
        # convert dates_s1 to datetime and extract month
        if self.date_in_weeks:
            dates = pd.Series(dates).dt.isocalendar().week % 12
            dates = torch.tensor(dates).long()
        else:
            dates = torch.tensor(dates.month).long()

        mask = mask.bool()
        dynamic_world = dynamic_world.long()
        # x = x.float()

        return (x, mask, dynamic_world, latlons, dates), label
