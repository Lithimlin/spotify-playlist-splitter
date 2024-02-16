from abc import ABC, abstractmethod
from functools import cached_property, reduce

import pandas as pd
import spotipy  # type: ignore[import-untyped]

FEATURE_COLUMNS = [
    "id",
    "duration_ms",
    "key",
    "tempo",
    "time_signature",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
]


def filter_metadata(track: dict) -> dict:
    """
    This function takes a dictionary representing a single track
    and returns a filtered dictionary containing only the specified keys.

    Parameters:
        track (dict): A dictionary representing a single track

    Returns:
        dict: A filtered dictionary containing only the specified keys
    """
    return {
        "id": track["id"],
        "name": track["name"],
        "album_name": track["album"]["name"],
        "album_id": track["album"]["id"],
        "artists": [artist["name"] for artist in track["artists"]],
        "release_date": track["album"]["release_date"],
        "popularity": track["popularity"],
    }


def get_artists_with_ids(track: dict) -> list[dict[str, str]]:
    """
    This function takes a dictionary representing a single track
    and returns a list of dicts mapping artists' names to their IDs.

    Parameters:
        track (dict): A dictionary representing a single track

    Returns:
        list[dict[str,str]]: A list of dicts mapping artists' names to their IDs
    """
    return [{"name": artist["name"], "id": artist["id"]} for artist in track["artists"]]


def divide_chunks(lst: list, n: int = 100) -> list[list]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class AbstractInfo(ABC):
    """
    This class is an abstract base class for all information classes about tracks in a Spotify collection.
    """

    def __init__(self, spotify: spotipy.Spotify):
        """
        Args:
            spotify (spotipy.Spotify): An authenticated Spotify API client.
        """
        self.spotify = spotify

    @cached_property
    @abstractmethod
    def length(self):
        pass

    @cached_property
    @abstractmethod
    def tracks(self):
        pass

    @property
    def tracks_ids(self):
        return [track["track"]["id"] for track in self.tracks]

    @cached_property
    def tracks_features(self):
        features = []
        for ids in divide_chunks(self.tracks_ids, 100):
            features += self.spotify.audio_features(ids)
        return features

    @property
    def _tracks_features_df(self):
        df = pd.DataFrame(self.tracks_features, columns=FEATURE_COLUMNS)
        df["duration_ms"] = (
            pd.to_timedelta(df["duration_ms"], unit="ms")
            .astype("timedelta64[s]")
            .astype("int64")
        )
        df = df.rename(columns={"duration_ms": "duration (s)"})
        return df

    @property
    def tracks_metadata(self):
        return [track["track"] for track in self.tracks]

    @cached_property
    def _artists_dataframe(self):
        artist_maps_list = list(
            reduce(
                lambda x, y: x + y, map(get_artists_with_ids, self.tracks_metadata), []
            )
        )
        df = pd.DataFrame(artist_maps_list).drop_duplicates()
        genres = []
        for ids in divide_chunks(df["id"].tolist(), 50):
            artists = self.spotify.artists(ids)["artists"]
            genres += [artist["genres"] for artist in artists]

        df["genres"] = genres
        return df

    @property
    def _artists_genre_dict(self):
        return pd.Series(
            self._artists_dataframe["genres"].values,
            index=self._artists_dataframe["name"],
        ).to_dict()

    @property
    def _tracks_metadata_df(self):
        df = pd.DataFrame(map(filter_metadata, self.tracks_metadata))
        df["release_date"] = pd.to_datetime(df["release_date"], format="mixed")

        # # As of 2024-01, the the albums API only returns empty genre lists, so this is useless...
        # genres = []
        # for ids in divide_chunks(df["album_id"].tolist(), 20):
        #     genres += [album["genres"] for album in self.spotify.albums(ids)["albums"]]
        # df["genres"] = genres

        # This is more inaccurate, but it's the best we can do since individual tracks don't have genres associated.
        df["genres"] = [
            frozenset(
                reduce(
                    lambda a, b: a + b,
                    [self._artists_genre_dict[artist] for artist in artists_list],
                    [],
                )
            )
            for artists_list in df["artists"]
        ]

        return df

    @cached_property
    def dataframe(self):
        return pd.merge(
            self._tracks_metadata_df, self._tracks_features_df, on="id", how="inner"
        )


class PlaylistInfo(AbstractInfo):
    """
    This class provides information about the tracks on a Spotify playlist.
    """

    def __init__(self, spotify: spotipy.Spotify, playlist_id: str):
        """
        Args:
            spotify (spotipy.Spotify): An authenticated Spotify API client.
            playlist_id (str): The Spotify ID for the playlist.
        """
        super().__init__(spotify=spotify)
        self.playlist_id = playlist_id

    @cached_property
    def length(self):
        return self.spotify.playlist_tracks(self.playlist_id, limit=1)["total"]

    @cached_property
    def tracks(self):
        tracks = []
        for offset in range(0, self.length, 100):
            tracks += self.spotify.playlist_tracks(
                self.playlist_id, offset=offset, limit=100
            )["items"]
        return tracks


class SavedInfo(AbstractInfo):
    def __init__(self, spotify: spotipy.Spotify):
        super().__init__(spotify=spotify)

    @cached_property
    def length(self):
        return self.spotify.current_user_saved_tracks(limit=1)["total"]

    @cached_property
    def tracks(self):
        tracks = []
        for offset in range(0, self.length, 50):
            tracks += self.spotify.current_user_saved_tracks(offset=offset, limit=50)[
                "items"
            ]
        return tracks
