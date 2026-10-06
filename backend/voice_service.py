import os

from faster_whisper import WhisperModel


MODEL_SIZE = "small"

_model = None


def get_model():

    global _model

    if _model is None:

        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

    return _model


def transcribe_audio(audio_file):

    temp_file = "temp_audio"

    with open(temp_file, "wb") as f:

        f.write(
            audio_file.getbuffer()
        )

    model = get_model()

    segments, info = model.transcribe(
        temp_file
    )

    text = ""

    for segment in segments:

        text += segment.text + " "

    try:

        os.remove(temp_file)

    except Exception:

        pass

    return text.strip()

