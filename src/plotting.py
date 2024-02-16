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
sns.set_style("whitegrid")


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
    fig.tight_layout(pad=0)
    n_cols, n_rows = (
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

    return images


def plot_stat_histograms(data: pd.DataFrame) -> plt.Figure:
    """Plot histograms for each column in the dataframe."""
    cols = get_clusterable_features(data)

    axes = (
        data.loc[:, cols]
        .rename(to_title_str, axis="columns")
        .hist(figsize=(20, 9), color=SPOTIFY_GREEN)
    )

    fig = axes[0][0].get_figure()
    fig.tight_layout()

    return fig


def plot_top_artists(data: pd.DataFrame, top_n=10) -> tuple[plt.Figure, plt.Axes]:
    """Plot the top artists in the dataframe."""
    artists = data["artists"].explode().value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.barplot(x=artists, y=artists.index, color=SPOTIFY_GREEN, orient="y", ax=ax)
    ax.set_title("Top artists", fontsize=14)
    ax.set_xlabel("Songs")
    ax.set_ylabel("Artists")

    fig.tight_layout()

    return fig, ax


def cluster_data(
    data: pd.DataFrame, k: int, features: Optional[list[str]] = None
) -> pd.DataFrame:
    """Cluster the data using K-means."""
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
    return data


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
