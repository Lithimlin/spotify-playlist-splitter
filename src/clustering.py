import base64

from io import BytesIO
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

SPOTIFY_GREEN = "#1DB954"
rcParams["font.family"] = ["Noto Sans CJK SC", "sans-serif"]
rcParams["figure.dpi"] = 200
sns.set_style("darkgrid")
sns.color_palette("dark")


def get_index_color(index: int):
    return sns.color_palette().as_hex()[index % len(sns.color_palette())]


def calc_squary_dimensions(n_plots: int):
    """Calculate the number of rows and columns for a grid of n_plots"""
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_cols, n_rows


def get_clusterable_features(data: pd.DataFrame) -> list[str]:
    """Get the columns that can be clustered."""
    return data.select_dtypes(exclude=["object"]).columns.to_list()


def convert_figure_to_str(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a string."""
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def to_title_str(text: str) -> str:
    return text.replace("_", " ").title()


def tile_figure_to_byte_images(fig: plt.Figure) -> list[str]:
    """Tile a matplotlib figure into a grid of images as byte streams."""
    FIGURE_PADDING = 1.5
    fig.tight_layout(pad=0, h_pad=FIGURE_PADDING, w_pad=FIGURE_PADDING)
    n_rows, n_cols = (
        fig.axes[0]
        .get_subplotspec()
        .get_topmost_subplotspec()
        .get_gridspec()
        .get_geometry()
    )
    image_size = fig.canvas.get_width_height()
    tile_width, tile_height = (image_size[0] // n_cols, image_size[1] // n_rows)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    # image = Image.frombytes("RGB", image_size, fig.canvas.tostring_rgb())

    grid = product(
        range(0, image_size[1], tile_height), range(0, image_size[0], tile_width)
    )

    images = []
    for y, x in grid:
        crop = image.crop((x, y, x + tile_width, y + tile_height))
        buf = BytesIO()
        crop.save(buf, format="png")
        images.append(base64.b64encode(buf.getbuffer()).decode("ascii"))
        # out_file = os.path.join("output", f"tile_{y}_{x}.png")
        # crop.save(out_file)

    return images


def plot_stat_histograms(data: pd.DataFrame, axsize=(4, 2)) -> plt.Figure:
    cols = get_clusterable_features(data)

    if "cluster" in cols:
        cols.remove("cluster")
        cluster = "cluster"
        color = None
    else:
        cluster = None
        color = SPOTIFY_GREEN

    fig, axes = plt.subplots(
        figsize=(axsize[0] * len(cols), axsize[1]), ncols=len(cols)
    )

    for col, ax in zip(cols, axes.flatten()):
        subsets = []
        if cluster:
            clusters = data[cluster].sort_values().unique()
            for value in clusters:
                subsets.append(data.loc[data[cluster] == value, col])
        else:
            subsets.append(data[col])

        ax.hist(subsets, stacked=True, color=color)
        ax.set_title(to_title_str(col), fontsize=14)

    fig.tight_layout()

    return fig


def plot_top_artists(data: pd.DataFrame, top_n=10, figsize=(20, 5)) -> plt.Figure:
    """Plot the top artists in the dataframe."""
    if "cluster" in data.columns:
        cols = ["artists", "cluster"]
        color = None
    else:
        cols = ["artists"]
        color = SPOTIFY_GREEN

    top_artists = (
        data["artists"]
        .explode()
        .value_counts()
        .reset_index()
        .head(top_n)["artists"]
        .to_list()
    )

    artists = data.loc[:, cols].explode("artists").value_counts().reset_index()
    artists = artists.loc[artists["artists"].isin(top_artists)]

    if "cluster" in cols:
        artists = artists.pivot(index="artists", columns="cluster", values="count")
        artists["count"] = artists.sum(axis=1)
        artists = artists.sort_index(ascending=False).sort_values(by="count")
        artists = artists.drop("count", axis=1)
    else:
        artists = (
            artists.set_index("artists")
            .sort_index(ascending=False)
            .sort_values(by="count")
        )

    fig, ax = plt.subplots(figsize=figsize)

    artists.plot.barh(ax=ax, color=color, width=0.85, stacked=True, legend=False)

    ax.set_title("Top Artists", fontsize=14)
    ax.set_xlabel("Songs")
    ax.set_ylabel(None)
    ax.grid(axis="y")

    fig.tight_layout()

    return fig


def plot_top_genres(data: pd.DataFrame, top_n=10, figsize=(20, 5)) -> plt.Figure:
    """Plot the top genres in the dataframe."""
    if "cluster" in data.columns:
        cols = ["genres", "cluster"]
        color = None
    else:
        cols = ["genres"]
        color = SPOTIFY_GREEN

    top_genres = (
        data["genres"].explode().value_counts().head(top_n).index.values.tolist()
    )

    genres = data.loc[:, cols].explode("genres").value_counts().reset_index()
    genres = genres.loc[genres["genres"].isin(top_genres)]

    if "cluster" in cols:
        genres = genres.pivot(index="genres", columns="cluster", values="count")
        genres["count"] = genres.sum(axis=1)
        genres = genres.sort_index(ascending=False).sort_values(by="count")
        genres = genres.drop("count", axis=1)
    else:
        genres = (
            genres.set_index("genres")
            .sort_index(ascending=False)
            .sort_values(by="count")
        )

    genres = genres.rename(to_title_str, axis="index")

    fig, ax = plt.subplots(figsize=figsize)
    genres.plot.barh(ax=ax, color=color, width=0.85, stacked=True, legend=False)

    ax.set_title("Top Genres", fontsize=14)
    ax.set_xlabel("Songs")
    ax.set_ylabel(None)
    ax.grid(axis="y")

    fig.tight_layout()

    return fig


def cluster_data(
    data: pd.DataFrame, k: int, features: Optional[list[str]] = None
) -> pd.DataFrame:
    """Cluster the data using K-means."""
    data = data.copy()
    if features is None:
        features = get_clusterable_features(data)
    date_cols = data.select_dtypes(include=["datetime64[ns]"]).columns.to_list()
    for col in date_cols:
        data[col] = data[col].view(int) // 10**9

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    kmeans = KMeans(init="k-means++", n_clusters=k, n_init="auto")
    kmeans.fit(scaled_data)
    data["cluster"] = kmeans.labels_

    clusters = pd.Series(kmeans.labels_, name="cluster")
    return clusters


def get_clustered_lists(
    data: pd.DataFrame, features: Optional[list[str]] = None
) -> list[pd.DataFrame]:
    """Get a list of DataFrames for each cluster."""
    if features is None:
        features = list()
    clusters = data["cluster"].sort_values().unique()

    data["duration"] = pd.to_timedelta(data["duration (s)"], unit="s")

    return [
        data.loc[data["cluster"] == i]
        .loc[:, ["id", "name", "artists", "album_name", "duration"] + features]
        .set_index("id")
        for i in clusters
    ]


def plot_clusters(
    data: pd.DataFrame, features: Optional[list[str]] = None
) -> tuple[plt.Figure, Axes3D]:
    """Plot the clusters for each given feature column."""
    if features is None:
        features = get_clusterable_features(data)
    if len(features) < 3:
        raise ValueError("3 features are required to plot clusters.")
    if len(features) > 3:
        features = features[:3]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(
        data[features[0]],
        data[features[1]],
        data[features[2]],
        c=data["cluster"],
        cmap="rainbow",
    )
    # ax.set_xlabel(features[0])
    # ax.set_ylabel(features[1])
    # ax.set_zlabel(features[2])
    # ax.set_title("Clusters")

    fig.tight_layout()

    return fig, ax
