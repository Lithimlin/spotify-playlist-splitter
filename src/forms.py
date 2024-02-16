# Flask Forms
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField
from wtforms.validators import InputRequired, Length

# https://www.digitalocean.com/community/tutorials/how-to-use-and-validate-web-forms-with-flask-wtf


class SplitForm(FlaskForm):
    k_clusters = IntegerField(
        "Number of Playlists", min_value=2, validators=[InputRequired()]
    )
    features = SelectField(id="feature")


class CommitForm(FlaskForm):
    name = StringField(
        "Playlist Name", validators=[InputRequired(), Length(min=1, max=100)]
    )
