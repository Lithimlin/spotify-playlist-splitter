# Flask Forms
from flask_wtf import FlaskForm
from markupsafe import Markup, escape
from wtforms import BooleanField, IntegerField, StringField
from wtforms.fields import Label
from wtforms.validators import InputRequired, Length, NumberRange
from wtforms.widgets import html_params

from src.clustering import to_title_str

# https://www.digitalocean.com/community/tutorials/how-to-use-and-validate-web-forms-with-flask-wtf


class Image():
    def __init__(self, image_src, **kwargs):
        self.image_src = image_src
        self.attributes = kwargs

    def __str__(self):
        return str(self())

    def __html__(self):
        return self()

    def __call__(self, image_src=None, **kwargs):
        kwargs = {**self.attributes, **kwargs}
        kwargs["src"] = image_src or self.image_src

        attributes = html_params(**kwargs)
        return Markup(f"<img {attributes} />")

    def __repr__(self):
        return f"<Image {self.image_src!r} {self.attributes!r}>"

class ImageLabel(Label):
    def __init__(self, *args, image, **kwargs):
        if "text" not in kwargs:
            kwargs["text"] = ""
        super().__init__(*args, **kwargs)
        self.image = image

    def __call__(self, text=None, image=None, image_kwargs=None, **kwargs):
        if "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        else:
            kwargs.setdefault("for", self.field_id)

        if image_kwargs is None:
            image_kwargs = {}

        attributes = html_params(**kwargs)
        text = escape(text or self.text)
        image = image or self.image
        return Markup(f"<label {attributes}>{text} {image(**image_kwargs)}</label>")

    def __repr__(self):
        return f"<ImageLabel {self.image!r} {self.text!r}>"


class ImageBooleanField(BooleanField):
    def __init__(self, *args, image_label: ImageLabel, **kwargs):
        super().__init__(*args, **kwargs)
        self.label: ImageLabel = image_label

def split_form_builder(features: list[str], histograms: list[str]) -> FlaskForm:
    class StaticSplitForm(FlaskForm):
        k_clusters = IntegerField(
            "Number of Playlists", validators=[InputRequired(), NumberRange(min=2)]
        )

        test_feature = ImageBooleanField(
            image_label=ImageLabel(
                field_id="test_feature",
                image=Image(
                    "https://flask-wtf.readthedocs.io/en/1.2.x/_static/flask-wtf-icon.png",
                    alt="wtf",
                    onload="this.width*=2;this.onload=null;",
                ),
            ),
            default=False
        )

    StaticSplitForm.dynamic_field = StringField("Dynamic Field")

    feature_labels = [
        ImageLabel(
            field_id=feature,
            # text=to_title_str(feature),
            image=Image(
                image_src="data:image/png;base64," + hist,
                alt=to_title_str(feature),
                onload="this.width/=2;this.onload=null;",
            ),
        )
        for feature, hist in zip(features, histograms)
    ]

    field_names = []

    for feature_label in feature_labels:
        setattr(
            StaticSplitForm,
            f"feature_{feature_label.field_id}",
            ImageBooleanField(image_label=feature_label, name=feature_label.field_id, default=False)
        )
        field_names.append(f"feature_{feature_label.field_id}")

    return StaticSplitForm(), field_names


class CommitForm(FlaskForm):
    name = StringField(
        "Playlist Name", validators=[InputRequired(), Length(min=1, max=100)]
    )
