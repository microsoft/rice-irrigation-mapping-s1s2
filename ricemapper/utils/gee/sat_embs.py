# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ee
import geopandas as gpd
import numpy as np
from datetime import date
from tqdm import tqdm


def flatten_sat_embeddings(
    gdf: gpd.GeoDataFrame, emb_column: str = "sat_embs", prefix: str = "emb_", drop=True
) -> gpd.GeoDataFrame:
    """
    Flatten satellite embedding arrays into separate columns.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with embedding column
        emb_column (str): Name of the column containing embedding arrays
        prefix (str): Prefix for the new column names

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with flattened embedding columns
    """
    result_gdf = gdf.copy()

    # Extract each dimension from the numpy arrays
    for i in range(64):
        col_name = f"{prefix}{i}"
        result_gdf[col_name] = result_gdf[emb_column].apply(
            lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else 0.0
        )

    if drop:
        result_gdf = result_gdf.drop(columns=[emb_column])

    return result_gdf


class SatelliteEmbeddingExtractor:
    """
    A class to extract satellite embeddings from Google Earth Engine's
    GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL collection.
    """

    def __init__(self, year: int, polys: gpd.GeoDataFrame):
        """
        Initialize the extractor with a specific year.

        Args:
            year (int): The year for which to extract embeddings
        """
        self.year = year
        self.start_date = date(year, 1, 1)
        self.end_date = date(year + 1, 1, 1)
        self.polys = polys

        # Ensure CRS is WGS84 (EPSG:4326) for Earth Engine
        if polys.crs != "EPSG:4326":
            print(f"Converting from {polys.crs} to EPSG:4326")
            self.polys = polys.to_crs("EPSG:4326")
        else:
            self.polys = polys

        # Initialize Google Earth Engine if not already done
        try:
            ee.Initialize()
        except Exception:
            # If initialization fails, try to authenticate first
            try:
                ee.Authenticate()
                ee.Initialize()
            except Exception as e:
                raise Exception(f"Failed to initialize Google Earth Engine: {e}")

        # No need for global bounding box since we filter by each polygon individually

        # Load the satellite embedding collection and filter by year
        # Keep as collection - we'll filter by each polygon's bounds individually
        self.image_collection = ee.ImageCollection(
            "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        ).filterDate(
            self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")
        )

        # Check if collection has any images
        if self.image_collection.size().getInfo() == 0:
            raise ValueError(f"No satellite embedding data found for year {year}")

    def extract_embeddings(self) -> gpd.GeoDataFrame:
        """
        Extract satellite embeddings for each polygon in the GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing polygons

        Returns:
            gpd.GeoDataFrame: Input GeoDataFrame with added 'sat_embs' column containing
                             64-dimensional embedding vectors
        """
        # Make a copy to avoid modifying the original
        result_gdf = self.polys.copy()

        # Initialize list to store embeddings
        embeddings_list = []

        for idx, row in tqdm(result_gdf.iterrows(), total=len(result_gdf)):
            try:
                # Get the geometry and convert to GEE format
                geometry = row.geometry

                # Convert shapely geometry to GEE geometry
                if geometry.geom_type == "Polygon":
                    # Get exterior coordinates
                    coords = list(geometry.exterior.coords)
                    ee_geometry = ee.Geometry.Polygon([coords])

                elif geometry.geom_type == "MultiPolygon":
                    # Handle MultiPolygon by taking the first polygon
                    coords = list(geometry.geoms[0].exterior.coords)
                    ee_geometry = ee.Geometry.Polygon([coords])
                else:
                    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

                # Extract embeddings for this polygon
                embedding_vector = self._extract_polygon_embedding(ee_geometry)
                embeddings_list.append(embedding_vector)

            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")
                # Add None or zeros for failed extractions
                embeddings_list.append(np.zeros(64))

        # Add embeddings as new column
        result_gdf["sat_embs"] = embeddings_list

        return result_gdf

    def _extract_polygon_embedding(self, ee_geometry: ee.Geometry) -> np.ndarray:
        """
        Extract embedding for a single polygon by computing mean over all pixels.

        Args:
            ee_geometry (ee.Geometry): Earth Engine geometry object

        Returns:
            np.ndarray: 64-dimensional embedding vector
        """
        # Filter the collection by this polygon's bounds to get the correct image
        polygon_image = self.image_collection.filterBounds(ee_geometry).first()

        if polygon_image is None:
            print(f"No image found for polygon bounds")
            return np.zeros(64)

        # Reduce the image to get mean values within the polygon
        reduced = polygon_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=ee_geometry, scale=10, maxPixels=1e9
        )

        # Get the computed values
        reduced_dict = reduced.getInfo()

        # Extract values for bands A00-A63
        band_names = [f"A{i:02d}" for i in range(64)]
        embedding_array = np.array(
            [
                (
                    reduced_dict.get(band, 0.0)
                    if reduced_dict.get(band, None) is not None
                    else 0.0
                )
                for band in band_names
            ]
        )

        return embedding_array

    def flatten_embeddings(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Flatten the sat_embs column into 64 separate columns (emb_0 to emb_63).

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame with 'sat_embs' column containing numpy arrays

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with 64 individual embedding columns
        """
        result_gdf = gdf.copy()

        # Extract each dimension from the numpy arrays
        for i in range(64):
            col_name = f"emb_{i}"
            result_gdf[col_name] = result_gdf["sat_embs"].apply(
                lambda x: x[i] if isinstance(x, np.ndarray) and len(x) > i else 0.0
            )

        return result_gdf

    def get_collection_info(self) -> dict:
        """
        Get information about the satellite embedding collection.

        Returns:
            dict: Information about the collection
        """
        try:
            # Get the first image to inspect its properties
            first_image = self.image_collection.first()
            band_names = first_image.bandNames().getInfo()
            collection_size = self.image_collection.size().getInfo()

            return {
                "year": self.year,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "collection_size": collection_size,
                "band_names": band_names,
                "num_bands": len(band_names),
            }
        except Exception as e:
            return {
                "year": self.year,
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "error": str(e),
            }
