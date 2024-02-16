from os import urandom
from datetime import timedelta

# Spotipy
import spotipy

# Flask
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    make_response,
)

# Flask Response Type
from werkzeug.wrappers import Response

# Session and Caching
from flask_session import Session
from flask_caching import Cache

# Pandas
from pandas import DataFrame

# Pydantic for Settings
from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Source Code
from src import clustering
from src.collect import PlaylistInfo, SavedInfo
from src import forms


class SpotifySettings(BaseSettings):
    """
    This class is used to store the configuration settings for the Spotify API.
    The settings are loaded from a .env file and can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="SPOTIFY_"
    )

    client_id: str
    """
    The client ID for the Spotify API.
    """

    client_secret: SecretStr
    """
    The client secret for the Spotify API.
    """

    redirect_uri: AnyUrl = Field(default="http://localhost:3002/")
    """
    The redirect URI for the Spotify API.
    """


SCOPE = "playlist-modify-public playlist-modify-private playlist-read-private user-library-read"
settings = SpotifySettings()

config = {
    "SECRET_KEY": urandom(64).hex(),
    "SESSION_TYPE": "filesystem",
    "SESSION_FILE_DIR": "./.flask_session",
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": timedelta(seconds=300).total_seconds(),
    # "CACHE_KEY_PREFIX": "flask_cache",
}
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)
Session(app)


def get_spotify() -> spotipy.Spotify | Response:
    """
    Returns a Spotify API client using the stored settings.
    If the stored access token is invalid, the user is redirected to the authorization page.

    Returns:
        spotipy.Spotify: A Spotify API client.
        Response: A redirect response to the root URL.
    """
    cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session)
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        redirect_uri=str(settings.redirect_uri),
        scope="playlist-modify-public playlist-modify-private playlist-read-private",
        cache_handler=cache_handler,
        show_dialog=True,
    )
    # Check if the stored access token is valid
    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        # If not, redirect to the authorization page
        return redirect("/")
    return spotipy.Spotify(auth_manager=auth_manager)


@app.route("/")
def index() -> Response:
    """
    This function handles the root URL of the application.
    It checks if the user is already authenticated, and if not,
    it redirects them to the authorization page.
    If the user is authenticated, it retrieves their user information
    from the Spotify API and displays it on the welcome page.

    Returns:
        Response: A redirect response to the root URL, the authorization page, or the welcome page.
    """
    cache_handler = spotipy.cache_handler.FlaskSessionCacheHandler(session)
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        client_id=settings.client_id,
        client_secret=settings.client_secret.get_secret_value(),
        redirect_uri=str(settings.redirect_uri),
        scope=SCOPE,
        cache_handler=cache_handler,
        show_dialog=True,
    )

    # Check if the user is already authenticated
    if request.args.get("code"):
        # If they are, retrieve their access token and refresh token
        auth_manager.get_access_token(code=request.args.get("code"))
        return redirect("/")

    if not auth_manager.validate_token(cache_handler.get_cached_token()):
        # If they are not, retrieve the authorization URL and display it in a template
        auth_url = auth_manager.get_authorize_url()
        return render_template("auth.html", auth_url=auth_url)

    # If the user is authenticated, retrieve their user information from the Spotify API
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    name = spotify.current_user()["display_name"]

    # Display the user's information on the welcome page
    return render_template("welcome.html", name=name)


@app.route("/sign_out")
def sign_out():
    """
    Handles the sign out button on the welcome page.

    Returns:
        Response: A redirect response to the root URL.
    """
    session.pop("token_info", None)
    return redirect("/")


@app.route("/playlists")
def playlists_page():
    spotify = get_spotify()
    page = request.args.get("page", 1, type=int)
    playlists = spotify.current_user_playlists(limit=50, offset=50 * (page - 1))
    return render_template("playlists.html", playlists=playlists, page=page)


@app.route("/playlists_json")
def playlists_json():
    spotify = get_spotify()
    page = request.args.get("page", 1, type=int)
    playlists = spotify.current_user_playlists(limit=50, offset=50 * (page - 1))
    return playlists


def analyze_songs(songs: DataFrame) -> str:
    """
    Plots histograms and top artists for the given songs dataframe.

    Args:
        songs (DataFrame): The songs dataframe.

    Returns:
        str: The HTML code for the histograms and top artists figures.
    """
    cols = clustering.get_clusterable_features(songs)

    hists = clustering.plot_stat_histograms(songs)

    hists = clustering.tile_figure_to_byte_images(hists)
    hists = [
        {"id": col, "data": hist, "title": clustering.to_title_str(col)}
        for col, hist in zip(cols, hists)
    ]

    artists = clustering.plot_top_artists(songs, figsize=(10, 5))
    genres = clustering.plot_top_genres(songs, figsize=(10, 5))

    return render_template(
        "analyze.html",
        histograms=hists,
        artists=clustering.convert_figure_to_str(artists),
        genres=clustering.convert_figure_to_str(genres),
        extra=dict(key="extra"),
    )


@app.route("/analyze", methods=["GET", "POST"])
@cache.cached(query_string=True)
def analyze_playlist():
    playlist_id = request.args.get("id")
    playlist = PlaylistInfo(spotify=get_spotify(), playlist_id=playlist_id)
    match request.method:
        case "GET":
            return analyze_songs(playlist.dataframe)

        case "POST":
            return validate_split_request(playlist.dataframe)


@app.route("/analyze_saved", methods=["GET", "POST"])
@cache.cached()
def analyze_saved_tracks():
    saved = SavedInfo(spotify=get_spotify())
    match request.method:
        case "GET":
            return analyze_songs(saved.dataframe)

        case "POST":
            return validate_split_request(saved.dataframe)


def validate_split_request(dataframe: DataFrame) -> Response:
    features = request.form.getlist("feature")
    if len(features) == 0:
        flash("Please select at least one feature.")
        print(request.url)
        return redirect(request.url)

    k_clusters = request.form.get("k_clusters")
    if k_clusters is None:
        flash("Please specify a number of playlists.")
        print(request.url)
        return redirect(request.url)

    # TODO: Send to split path for sane caching
    # Put dataframe in request JSON?
    return split_playlist(dataframe, int(k_clusters), features)


def split_list(songs: DataFrame, features: list[str]) -> Response:
    cols = clustering.get_clusterable_features(songs)

    hists = clustering.plot_stat_histograms(songs)

    hists = clustering.tile_figure_to_byte_images(hists)
    hists = [
        {"data": hist, "title": clustering.to_title_str(col)}
        for col, hist in zip(cols, hists)
    ]

    artists = clustering.plot_top_artists(songs, figsize=(10, 5))
    genres = clustering.plot_top_genres(songs, figsize=(10, 5))

    def format_duration(duration: timedelta) -> str:
        minutes = int(duration.total_seconds() // 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{minutes:02d}:{seconds:02d}"

    lists: list[dict] = []

    for i, playlist in enumerate(clustering.get_clustered_lists(songs, features)):
        features_stats = {
            feature: (
                round(playlist[feature].min(), 2),
                round(playlist[feature].max(), 2),
            )
            for feature in features
        }
        playlist = playlist.drop(columns=features)
        playlist_dict = {
            "html": playlist.to_html(
                border=0,
                classes="playlist",
                index=False,
                formatters={
                    "artists": lambda x: ", ".join(x),
                    "duration": format_duration,
                },
            ),
            "color": clustering.get_index_color(i),
            "features": features_stats,
            "id": i,
        }
        lists.append(playlist_dict)

    return render_template(
        "split.html",
        lists=lists,
        histograms=hists,
        artists=clustering.convert_figure_to_str(artists),
        genres=clustering.convert_figure_to_str(genres),
    )


@app.route("/split", methods=["POST"])
def split_playlist(songs: DataFrame, k_clusters: int, features: list[str]) -> Response:
    cache.delete_memoized(analyze_playlist)
    cache.delete_memoized(analyze_saved_tracks)
    songs["cluster"] = clustering.cluster_data(songs, k_clusters, features)

    return split_list(songs, features)

    # return songs.to_html()
    # return render_template("split.html")


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run using 'python'. Use 'flask run' instead!"
    )
