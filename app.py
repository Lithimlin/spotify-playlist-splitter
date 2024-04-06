from datetime import timedelta
from functools import wraps
from itertools import chain
from os import urandom
from random import shuffle

# Spotipy
import spotipy
# Flask
from flask import (Flask, abort, current_app, flash, redirect, render_template,
                   request, session)
# Session and Caching
from flask_caching import Cache
from flask_session import Session
# Pandas
from pandas import DataFrame
# Pydantic for Settings
from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
# Flask Response Type
from werkzeug.wrappers import Response

# Source Code
from src import clustering
from src.collect import PlaylistInfo, SavedInfo


class AuthenticationError(RuntimeError):
    """
    Raised when the user is not authenticated.
    """

    def __init__(self, message: str, redirect_url: str = "/"):
        super().__init__(message)
        self.redirect_url = redirect_url


def debug_only(func):
    """
    Decorator that only allows the function to be called in debug mode.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_app.debug:
            abort(404)
        return func(*args, **kwargs)

    return wrapper


class SpotifySettings(BaseSettings):
    """
    This class is used to store the configuration settings for the Spotify API.
    The settings are loaded from a .env file and can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="SPOTIFY_", extra='ignore'
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


def get_spotify() -> spotipy.Spotify:
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
        raise AuthenticationError("User is not authenticated")
    return spotipy.Spotify(auth_manager=auth_manager)


@app.route("/")
def index() -> str | Response:
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
def playlists_page() -> str | Response:
    """
    Handles the playlists page of the application.
    It retrieves the user's playlists from the Spotify API, and displays them in a pagination.

    Returns:
        str: The HTML code for the playlists page.
        Response: A redirect response if authentication failed.
    """
    try:
        spotify = get_spotify()
    except AuthenticationError as ae:
        flash(str(ae), category="error")
        return redirect(ae.redirect_url)

    page = request.args.get("page", 1, type=int)
    playlists = spotify.current_user_playlists(limit=50, offset=50 * (page - 1))
    return render_template("playlists.html", playlists=playlists, page=page)


@app.route("/playlists_json")
@debug_only
def playlists_json():
    """
    A debug page to inspect the playlists returned by the Spotify API.

    Returns:
        Response: A JSON response containing the playlists.
    """
    try:
        spotify = get_spotify()
    except AuthenticationError as ae:
        flash(str(ae), category="error")
        return redirect(ae.redirect_url)

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
    features = clustering.get_clusterable_features(songs)

    hists_figure = clustering.plot_stat_histograms(songs)

    hists_bytes = clustering.tile_figure_to_byte_images(hists_figure)
    hists_dict = [
        {"id": feature, "data": histogram, "title": clustering.to_title_str(feature)}
        for feature, histogram in zip(features, hists_bytes)
    ]

    artists = clustering.plot_top_artists(songs, figsize=(10, 5))
    genres = clustering.plot_top_genres(songs, figsize=(10, 5))

    return render_template(
        "analyze.html",
        histograms=hists_dict,
        artists=clustering.convert_figure_to_str(artists),
        genres=clustering.convert_figure_to_str(genres),
        extra=dict(key="extra"),
    )


@app.route("/analyze", methods=["GET"])
# @cache.cached(query_string=True)
def analyze_playlist() -> str | Response:
    """
    Handles the analysis page for playlists.
    It retrieves songs from a given playlist ID, analyzes them, and displays the results.

    Returns:
        str: The HTML code for the analysis page.
        Response: A redirect response if authentication failed.
    """
    try:
        spotify = get_spotify()
    except AuthenticationError as ae:
        flash(str(ae), category="error")
        return redirect(ae.redirect_url)

    playlist_id = request.args.get("id")

    if session.get("list_id") == playlist_id:
        playlist_frame = DataFrame(session["songs_dict"])
        return analyze_songs(playlist_frame)

    playlist = PlaylistInfo(spotify=spotify, playlist_id=playlist_id)

    session["songs_dict"] = playlist.dataframe.to_dict("list")
    session["list_id"] = playlist_id

    return analyze_songs(playlist.dataframe)


@app.route("/analyze_saved", methods=["GET"])
# @cache.cached()
def analyze_saved_tracks() -> str | Response:
    """
    Handles the analysis page for saved tracks.
    It retrieves songs from the user's saved tracks, analyzes them, and displays the results.

    Returns:
        str: The HTML code for the analysis page.
        Response: A redirect response if authentication failed.
    """
    try:
        spotify = get_spotify()
    except AuthenticationError as ae:
        flash(str(ae), category="error")
        return redirect(ae.redirect_url)

    if session.get("list_id") == "saved":
        saved_frame = DataFrame(session["songs_dict"])
        return analyze_songs(saved_frame)

    saved = SavedInfo(spotify=spotify)

    session["songs_dict"] = saved.dataframe.to_dict("list")
    session["list_id"] = "saved"

    return analyze_songs(saved.dataframe)


def validate_split_request() -> bool:
    """
    Validates the split request.

    Returns:
        bool: True if the request is valid, False otherwise.
    """
    features = request.form.getlist("feature")
    if len(features) == 0:
        flash("Please select at least one feature.")
        return False

    k_clusters = request.form.get("k_clusters")
    if k_clusters is None:
        flash("Please specify a number of playlists.")
        return False

    return True


def show_split_list(songs: DataFrame, features: list[str]) -> str:
    """
    Displays the clustered playlists.
    First lists the clustered playlists with their songs,
    then displays the histograms, top artists, and top genres.

    Args:
        songs (DataFrame): The songs dataframe.
        features (list[str]): The features that were used for clustering.

    Returns:
        str: The HTML code for the split page.
    """
    cols = clustering.get_clusterable_features(songs)

    hists_figure = clustering.plot_stat_histograms(songs)

    hists_bytes = clustering.tile_figure_to_byte_images(hists_figure)
    hists_dict = [
        {"data": histogram, "title": clustering.to_title_str(feature)}
        for feature, histogram in zip(cols, hists_bytes)
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
        histograms=hists_dict,
        artists=clustering.convert_figure_to_str(artists),
        genres=clustering.convert_figure_to_str(genres),
    )


@app.route("/split", methods=["POST"])
def split_playlist() -> str | Response:
    """
    Handles the split page for playlists.
    It validates split requests,
    clusters the songs in a given playlist,
    then displays the lists, histograms, top artists, and top genres.

    Returns:
        str: The HTML code for the split page.
        Response: A redirect response if form validation failed,
            the user re-clustered, or shuffled a playlist.
    """
    if session.get("invalid_form") == "commit":
        session["invalid_form"] = None

        clusters = DataFrame(chain.from_iterable(session["clusters"]))
        return show_split_list(clusters, session["features"])

    if request.form.get("commit"):
        return redirect("/commit", code=307)

    if request.form.get("shuffle"):
        cluster_id = int(request.form["shuffle"])
        shuffle(session["clusters"][cluster_id])
        session.modified = True

        clusters = DataFrame(chain.from_iterable(session["clusters"]))
        return show_split_list(clusters, session["features"])

    if request.form.get("recluster"):
        k_clusters = session["k_clusters"]
        features = session["features"]

    else:
        if not validate_split_request():
            playlist_id = session["list_id"]
            if playlist_id == "saved":
                return redirect("/analyze_saved")
            return redirect(f"/analyze?id={playlist_id}")

        k_clusters = int(request.form["k_clusters"])
        features = request.form.getlist("feature")

    songs = DataFrame(session["songs_dict"])

    songs["cluster"] = clustering.cluster_data(songs, k_clusters, features)

    clusters_list = [
        songs.loc[songs["cluster"] == cluster_id].to_dict("records")
        for cluster_id in range(k_clusters)
    ]

    session["clusters"] = clusters_list
    session["features"] = features
    session["k_clusters"] = k_clusters

    return show_split_list(songs, features)


def create_playlist(name: str, songs: list[str]) -> None:
    """
    Creates a playlist with the given name and songs.

    Args:
        name (str): The name of the playlist.
        songs (list[str]): The IDs of the songs to add to the playlist.
    """
    spotify = get_spotify()

    playlist = spotify.user_playlist_create(
        user=spotify.me()["id"],
        name=name,
        public=False,
        collaborative=False,
        description="Created by the Playlist Splitter",
    )

    spotify.user_playlist_add_tracks(
        user=spotify.me()["id"], playlist_id=playlist["id"], tracks=songs
    )


@app.route("/commit", methods=["POST"])
def commit_playlist() -> Response:
    """
    Handles the commit page for playlists.
    It creates playlists with the given names and songs on a valid form.

    Returns:
        Response: A redirect response if authentication or form validation failed,
            or the playlists were created successfully.
    """
    names = {
        int(key.replace("name_", "")): name
        for key, name in request.form.items()
        if key.startswith("name_")
    }

    for name in names.values():
        if name == "":
            flash("Please specify a name for each playlist.", category="error")
            session["invalid_form"] = "commit"
            return redirect("/split", code=307)

    clusters = session["clusters"]

    playlists = {
        name: [cluster["id"] for cluster in clusters[idx]]
        for idx, name in names.items()
    }

    try:
        for name, songs in playlists.items():
            create_playlist(name, songs)
    except AuthenticationError as ae:
        flash(f"Could not authenticate with Spotify: {ae}", category="error")
        session["invalid_form"] = "commit"
        return redirect(ae.redirect_url)

    session.pop("clusters", None)
    session.pop("features", None)
    session.pop("k_clusters", None)
    session.pop("songs_dict", None)
    session.pop("list_id", None)

    return redirect("/playlists")


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run using 'python'. Use 'flask run' instead!"
    )
