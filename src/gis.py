"""
This module contains source code to analyze geographical information (gis) data.
It includes two classes where on inherits from the other. The class purposes are especially suited for
visualization of the data

Furthermore, ut contains methods to generate features out of Geographical Information System (GIS).
Note that all of those classes are using the BaseEstimator and TransformerMixin classes from the scikit-learn bases classes.
In order to use the ``fit_transform`` implemented by the scikit-learn, every single Feature Transformer needs to have a  ``fit`` and ``transform`` method.
The ``fit`` methods returns the object itself and is therefore not documented
"""
from __future__ import annotations
from typing import Union, Sequence, List, Mapping

import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN
import numpy as np
from pandas import DataFrame

# temporary hack to enable relative imports when script is not run within package
# see https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time?rq=1
# or https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py?rq=1
try:
    from utilities import Logger
    from baseattrs import BaseStats
except ImportError:
    from .utilities import Logger
    from .baseattrs import BaseStats

log = Logger.initialize_log()

try:
    from hdbscan import HDBSCAN
except ImportError:
    log.warning(
        "HDBSCAN is not part of the scikit-learn package. To use it you need to install HDBSCAN via pip install HDBSCAN"
    )
try:
    import folium
    from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster
except ImportError:
    log.warning(
        f"Failed to import folium package. Please check if it installed inside the Python environment. Cannot use plotting geo data on real world maps."
    )

__all__ = ["GisAnalyzer",
           "GisAnalyzerWithClusterLabel",
           "GisBasicTransformer",
           "GisDistance",
           "GisCluster",
           ]


class GisAnalyzer(BaseStats):
    """
    A class used to visualise geographical data based on latitude and longitude columns
    """

    def __init__(self, data: Union[str, Sequence[str]], lat: str = "", lon: str = ""):
        """
        Initializes a GIS Analyzer object. It expects a DataFrame as input. Furthermore, it has the ``lat`` and ``lon`` columns which are
        used for defining the latitude and longitude. They do not have to be set upon object creation but can be reset using the corresponding
        ``set_lat`` and ``set_lon`` methods.
        It also has three mapping objects indicating the map style, a color list and a grad list.

        :param data: the data holding the geographical latitude and longitude columns
        :type data: pd.DataFrame
        :param lat: the latitude feature column name
        :type lat: str
        :param lon: the longitude feature column name
        :type lon: str
        """
        super().__init__(data)
        self.lat = lat
        self.lon = lon
        self._style_list = [
            "OpenStreetMap",
            "Mapbox Bright",
            "Mapbox Control Room",
            "Stamen Terrain",
            "Stamen Toner",
            "Stamen Watercolor",
            "CartoDB positron",
            "CartoDB dark_matter",
        ]
        self._color_list = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "lightred",
            "beige",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "pink",
            "lightblue",
            "lightgreen",
            "gray",
            "black",
            "lightgray",
        ]
        self._grad_list = {0.2: "blue", 0.4: "lime", 0.6: "orange", 1: "red"}
        self.map = None

    # TODO implement class methods to get lat,lon from other types than pandas

    def set_lat(self, lat: str) -> GisAnalyzer:
        """
        Resets the name of the latitude column

        :param lat: the label oft the latitude columns
        :type lat: str
        :return: the object with its ``lat`` attribute reset
        """
        self.lat = lat
        return self

    def set_lon(self, lon: str) -> GisAnalyzer:
        """
        Resets the name of the longitude column

        :param lon: the label oft the longitude columns
        :type lon: str
        :return: the object with its ``lon`` attribute reset
        """
        self.lon = lon
        return self

    @property
    def initial_lat(self) -> float:
        """
        A property yielding the first latitude entry.

        :return: the first entry of the latitude column
        """
        return self.data.loc[0, self.lat]

    @property
    def initial_lon(self) -> float:
        """
        A property yielding the first longitude entry

        :return: the first entry of the longitude column
        """
        return self.data.loc[0, self.lat]

    @property
    def coordinates(self) -> List[str]:
        """
        A property putting the latitude and longitude to a list. Used as an alias to avoid writing [lat,lon] any time the GIS is plotted

        :return: a list of the latitude and longitude
        """
        return [self.lat, self.lon]

    def _generate_base_map(self, style: int, zoom_start: int) -> GisAnalyzer:
        """
        Private method that generates a basic BaseMap and stores it in OBJECT.map
        """
        self.map = folium.Map(
            location=[self.initial_lat, self.initial_lon],
            control_scale=True,
            zoom_start=zoom_start,
            tiles=self._style_list[style],
        )
        return self

    def generate_heat_map(self, style: int = 6, zoom_start: int = 12):
        """
        Creates a HeatMap with the coordinates found in the latitude and longitude columns.

        :param style: indicates which style of the map to be used
        :type style: int
        :param zoom_start: set the initial zoom of the map
        :type zoom_start: int
        :return:
        """
        self._generate_base_map(style, zoom_start)
        log.info("Creating a HeatMap and saving it in the .map attribute.")
        HeatMap(
            data=self.data[self.coordinates]
            .groupby(self.coordinates)
            .sum()
            .reset_index()
            .values.tolist(),
            radius=8,
            max_zoom=13,
        ).add_to(self.map)
        return self.map

    def generate_heat_map_with_time(
        self,
        style: int = 6,
        zoom_start: int = 12,
        time_var: str = "hours",
        time_max_val: int = 24,
    ):
        """
        Generates a interactive HeatMap which enables a geographic visualization by time.

        :param style: indicates which style of the map to be used
        :type style: int
        :param zoom_start: set the initial zoom of the map
        :type zoom_start: int
        :param time_var: the label of the time column by which the data is analyzed by
        :type time_var: str
        :param time_max_val: the maximal time value to be shown
        :type time_max_val: int
        :return:
        """
        junk_coord = (
            -360
        )  # Junk values are inserted here such that the time variable matches the time filter
        self._generate_base_map(style, zoom_start)
        df_tmp = self.data[[self.lat, self.lon, time_var]]
        time_list = [
            [[junk_coord, junk_coord]]
        ] * time_max_val  # Initial list with junk values to be replaced
        for t in df_tmp[time_var].sort_values().unique():
            if t > 0:
                time_list[t - 1] = (
                    df_tmp.loc[df_tmp[time_var] == t, self.coordinates]
                    .groupby(self.coordinates)
                    .sum()
                    .reset_index()
                    .values.tolist()
                )
            else:
                time_list[time_max_val] = (
                    df_tmp.loc[df_tmp[time_var] == t, self.coordinates]
                    .groupby(self.coordinates)
                    .sum()
                    .reset_index()
                    .values.tolist()
                )
        HeatMapWithTime(
            time_list,
            radius=5,
            gradient=self._grad_list,
            min_opacity=0.5,
            max_opacity=0.8,
            use_local_extrema=True,
        ).add_to(self.map)

    def generate_map_cluster(
        self, style=6, zoom_start=12, include_heat_map: bool = True
    ):
        """
        Generates a map which includes cluster depending on the GIS distances between the different points. The Cluster itself depends on the zooming level

        :param style: indicates which style of the map to be used
        :type style: int
        :param zoom_start: set the initial zoom of the map
        :type zoom_start: int
        :param include_heat_map: a flag which indicates if an additional HeatMap shall be added to the Map
        :type include_heat_map: bool
        :return:
        """
        self.generate_heat_map(
            style=style, zoom_start=zoom_start
        ) if include_heat_map else self._generate_base_map(style, zoom_start)
        self.map.add_child(
            MarkerCluster(locations=list(zip(self.data[self.lat], self.data[self.lon])))
        )

    def simple_plot(self, **options: Mapping[str, ]):
        """
        Generates a simple scatter plot of the coordinates. The scatter is an abstraction of the data by not include the actual map.
        It can be used to quickly have a look on a basic map.

        :param options: the seaborn scatter plot options
        :return:
        """
        sns.scatterplot(self.data[self.lat], self.data[self.lon], **options)


class GisAnalyzerWithClusterLabel(GisAnalyzer):
    """
    A class derived from the GisAnalyzer used to visualise geographical data based on latitude and longitude columns and take into account cluster labels
    """

    def __init__(
        self,
        data: Union[str, DataFrame],
        lat: str = "",
        lon: str = "",
        cluster_label: Union[str, Sequence] = None,
    ):
        """
        Initializes a GIS Analyzer object. It expects a DataFrame as input. Furthermore, it has the ``lat`` and ``lon`` columns which are
        used for defining the latitude and longitude. They do not have to be set upon object creation but can be reset using the corresponding
        ``set_lat`` and ``set_lon`` methods. Additionally, a set of labels can be indicated as well. The purpose hereby lies especially in visualizing
        geographic clusters.
        It also has three mapping objects indicating the map style, a color list and a grad list.

        :param data: the data holding the geographical latitude and longitude columns
        :type data: pd.DataFrame
        :param lat: the latitude feature column name
        :type lat: str
        :param lon: the longitude feature column name
        :type lon: str
        :param cluster_label: single or multiple column labels indicating the clustering label
        :type cluster_label: list
        """
        super().__init__(data, lat, lon)
        if cluster_label is None:
            cluster_label = []
        self.cluster_label = (
            cluster_label if isinstance(cluster_label, list) else [cluster_label]
        )

    def simple_plot_by_cluster_labels(self, **options: Mapping[str, ]):
        """
        Plots the geographical data by all the different cluster labels.

        :param options: options used by the ``simple_plot``
        :type options: dict
        :return:
        """
        # TODO add options into the plot
        if self.cluster_label:
            for label in self.cluster_label:
                self.simple_plot(**{"hue": label})


class _BaseClass:
    """
    Defines basic attributes that each GIS analyzing class has.
    """

    def __init__(self, lat: str, lon: str):
        """
        Defines the basis feature of each GIS analyzing class. These include:

            - Latitude: a geo point's latitude
            - Longitude: a geo point's longitude
            - Coordinates: in the understanding of this class, the coordinates attribute simply puts the latitude and longitude attributes in a list

        :param lat: the column name of the latitude column
        :type lat: str
        :param lon: the column name of the longitude column
        :type lon: str
        """
        self.lat = lat
        self.lon = lon
        self.coordinates = [lat, lon]


class GisBasicTransformer(_BaseClass, BaseEstimator, TransformerMixin):
    """
    Derives primitive feature transformation for GIS
    The primitive transformations are:

    - ``round``: round the GIS to a specific digit length. This can be considered as a feature as it groups together points with similar values.
      It reduces information but reduces the dimension of the feature

    - ``radians``: a simple transformation to get the radians of the GIS.

    """
    # TODO include x,y,z features.
    def __init__(self, lat: str, lon: str, round_factor: int = 0, radian: bool = True):
        """
        Initializes a primitive feature creator object

        :param lat: the column name of the latitude column
        :type lat: str
        :param lon: the column name of the longitude column
        :type lon: str
        :param round_factor: if set and >0, it indicate the number of digits to round the geo data.
        :type round_factor: int
        :param radian: a flag indicating if the radians shall be derived
        :type radian: bool
        """
        super(GisBasicTransformer).__init__(lat, lon)
        self.round = round_factor
        self.radian = radian
        self.feature_names = []

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transforms the DataFrame. Checks the both arguments and applies the respective function if the condition hols (round >0; radian = True)

        :param X: the original DataFrame
        :return:
        """
        for coord in self.coordinates:
            if self.round:
                X[coord + "_round"] = X[coord].round(self.round)
            if self.radian:
                X[coord + "_radian"] = X[coord].apply(lambda x: np.radians(x))
        self.feature_names = X.columns.tolist()
        return X


class GisDistance(_BaseClass, BaseEstimator, TransformerMixin):
    """
    Calculates the haversine distance between two geo points
    """

    def __init__(
        self, lat_1: str, lon_1: str, lat_2: str, lon_2: str, label: str = "distance"
    ):
        """
        Initializes an object which calculates the Haversine distance between two points

        :param lat_1: the column name of the first point's latitude column
        :type lat_1: str
        :param lon_1: the column name of the first point's longitude column
        :type lon_1: str
        :param lat_2: the column name of the second point's latitude column
        :type lat_2: str
        :param lon_2: the column name of the second point's longitude column
        :type lon_2: str
        :param label: the column label of the newly created column capturing the distance
        :type label: str
        """
        super().__init__(lat_1, lon_1)
        self.lat_2 = lat_2
        self.lon_2 = lon_2
        self.label = label
        self.feature_names = []

    def fit(self, X, y=None, **kwargs):
        return self

    def haversine_distance(self, row):
        """
        The formulae for the haversine distance. It takes a DataFrame's row as an input arguments, derives the four GIS points and applies the Formula to it.

        :param row: a DataFrame row
        :type row: pd.DataFrame
        :return: the haversine distance for the row
        """
        lat_1, lon_1 = row[self.lat], row[self.lon]
        lat_2, lon_2 = row[self.lat_2], row[self.lon_2]
        radius = 6371  # km#
        d_lat = np.radians(lat_2 - lat_1)
        d_lon = np.radians(lon_2 - lon_1)
        a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + np.cos(np.radians(lat_1)) * np.cos(
            np.radians(lat_2)
        ) * np.sin(d_lon / 2) * np.sin(d_lon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c
        return distance

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Applies the haversine_distance formula to all rows and saves the value in a newly generated column

        :param X: the original DataFrame
        :return:
        """
        X[self.label] = X.apply(self.haversine_distance, axis=1)
        self.feature_names = X.columns.tolist()
        return X


class GisCluster(_BaseClass, BaseEstimator, TransformerMixin):
    """
    Applies Machine Learning based clustering on the GIS data.
    """

    def __init__(
        self,
        lat: str,
        lon: str,
        kmeans: bool = True,
        kmeans_params: Mapping[str, ] = None,
        dbscan: bool = True,
        dbscan_params: Mapping[str, ] = None,
        birch: bool = True,
        birch_params: Mapping[str, ] = None,
        hdbscan: bool = True,
        hdbscan_params: Mapping[str, ] = None,
        agglomerative: bool = True,
        agglomerative_params: Mapping[str, ] = None,
    ):
        """
        Creates an object that applies a possibly multitude of different clustering algorithms on the data

        :param lat: the column name of the latitude column
        :type lat: str
        :param lon: the column name of the longitude column
        :type lon: str
        :param kmeans: a flag indicating if the k-means clustering algorithm is applied
        :type kmeans: bool
        :param kmeans_params: indicates k-means clustering hyper parameters
        :type kmeans_params: dict
        :param dbscan:  a flag indicating if the DBSCAN clustering algorithm is applied
        :type dbscan: bool
        :param dbscan_params: indicates DBSCAN clustering hyper parameters
        :type dbscan_params: dict
        :param birch: a flag indicating if the BIRCH clustering algorithm is applied
        :type birch:  bool
        :param birch_params: indicates BIRCH clustering hyper parameters
        :type birch_params: dict
        :param hdbscan: a flag indicating if the HDBSCAN clustering algorithm is applied
        :type hdbscan:  bool
        :param hdbscan_params: indicates HDBSCAN clustering hyper parameters
        :type hdbscan_params: dict
        :param agglomerative: a flag indicating if the Agglomerative clustering algorithm is applied
        :type agglomerative:  bool
        :param agglomerative_params: indicates Agglomerative clustering hyper parameters
        :type agglomerative_params: dict
        """

        super().__init__(lat, lon)

        # algorithm attributes
        self.kmeans = kmeans
        self.dbscan = dbscan
        self.birch = birch
        self.hdbscan = hdbscan
        self.agglomerative = agglomerative
        self.feature_names = []

        # algorithm params initialization
        if kmeans_params is None:
            kmeans_params = {}
        self.kmeans_params = kmeans_params

        if dbscan_params is None:
            dbscan_params = {}
        self.dbscan_params = dbscan_params

        if birch_params is None:
            birch_params = {}
        self.birch_params = birch_params

        if hdbscan_params is None:
            hdbscan_params = {}
        self.hdbscan_params = hdbscan_params

        if agglomerative_params is None:
            agglomerative_params = {}
        self.agglomerative_params = agglomerative_params

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Applies the clustering based on which algorithm with respective parameters have been specified.
        For each clustering algorithm set to True, the clustering is performed using the respective algorithm's hyper parameters

        :param X: The initial DataFrame
        :return:
        """
        if self.kmeans:
            cluster = KMeans(**self.kmeans_params)
            X["clstr_kmeans"] = cluster.fit_predict(X[self.coordinates])
        if self.dbscan:
            cluster = DBSCAN(**self.dbscan_params)
            X["clstr_dbscan"] = cluster.fit_predict(X[self.coordinates])
        if self.hdbscan:
            try:
                cluster = HDBSCAN(**self.hdbscan_params)
                X["clstr_hdbscan"] = cluster.fit_predict(X[self.coordinates])
            except NameError:
                log.error("HDBSCAN was not found in the module and cannot be used")
                pass
        if self.birch:
            cluster = Birch(**self.birch_params)
            X["clstr_birch"] = cluster.fit_predict(X[self.coordinates])
        if self.agglomerative:
            cluster = AgglomerativeClustering(**self.agglomerative_params)
            X["clstr_aggl"] = cluster.fit_predict(X[self.coordinates])
        self.feature_names = X.columns.tolist()
        return X


# TODO Revers geo coding
# TODO x,yz  features
