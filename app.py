import streamlit as st

from backend.adviser import AgricultureAdviser
from backend.image_service import analyze_image
from backend.voice_service import transcribe_audio


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌱",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================
# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #F7FBF4;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 18px;
        color: #5F6F52;
        margin-bottom: 25px;
    }

    /* Feature cards */
    .feature-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #C8E6C9;
        margin-bottom: 15px;
        background-color: #FFFFFF;
        color: #355E3B;
    }

    /* Feature card headings */
    .feature-card b {
        color: #2E7D32;
    }

    /* Answer box */
    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #A5D6A7;
        background-color: #F1F8E9;
        color: #263D26;
    }

    /* Section headings */
    h3 {
        color: #33691E;
    }

    /* Primary button */
    .stButton > button {
        background-color: #388E3C;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #2E7D32;
        color: white;
    }

    /* Text area */
    textarea {
        border: 1px solid #A5D6A7 !important;
        border-radius: 10px !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-color: #A5D6A7;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# INITIALIZE ADVISER
# ==================================================

@st.cache_resource
def load_adviser():

    return AgricultureAdviser()


adviser = load_adviser()


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🌱 AI Smart Agriculture</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Your AI-powered agriculture adviser for practical farming guidance.
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# AGRICULTURE TOPICS
# ==================================================

st.markdown("### How can I help you?")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="feature-card">
        🌾 <b>Crop Management</b><br>
        Crop selection and cultivation guidance.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
        🌱 <b>Soil & Nutrition</b><br>
        Soil, fertilizer and nutrient guidance.
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
        💧 <b>Irrigation</b><br>
        Water management and irrigation advice.
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="feature-card">
        🐛 <b>Pest & Disease</b><br>
        Identify possible crop problems and management.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown("### Ask your agriculture question")

question = st.text_area(
    "Type your question",
    placeholder=(
        "Example: What are the major pests of rice?"
    ),
    height=120,
    label_visibility="collapsed"
)


# ==================================================
# IMAGE UPLOAD
# ==================================================

image_file = st.file_uploader(
    "📷 Upload a crop image (optional)",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==================================================
# VOICE UPLOAD
# ==================================================

audio_file = st.file_uploader(
    "🎤 Upload a voice question (optional)",
    type=[
        "wav",
        "mp3",
        "m4a",
        "ogg"
    ]
)


# ==================================================
# ASK BUTTON
# ==================================================

ask_button = st.button(
    "🌱 Get Agriculture Advice",
    type="primary",
    use_container_width=True
)


# ==================================================
# PROCESS REQUEST
# ==================================================

if ask_button:

    if not question and not image_file and not audio_file:

        st.warning(
            "Please enter a question, upload an image, "
            "or upload a voice question."
        )

        st.stop()

    # ----------------------------------------------
    # Voice transcription
    # ----------------------------------------------

    voice_text = ""

    if audio_file:

        with st.spinner(
            "Understanding your voice..."
        ):

            voice_text = transcribe_audio(
                audio_file
            )

    # ----------------------------------------------
    # Combine text + voice
    # ----------------------------------------------

    final_question = question.strip()

    if voice_text:

        if final_question:

            final_question = (
                f"{final_question}\n\n"
                f"Voice question: {voice_text}"
            )

        else:

            final_question = voice_text

    # ----------------------------------------------
    # Image analysis
    # ----------------------------------------------

    image_description = ""

    if image_file:

        st.image(
            image_file,
            caption="Uploaded crop image",
            use_container_width=True
        )

        with st.spinner(
            "Analyzing the crop image..."
        ):

            image_description = analyze_image(
                image_file
            )

    # ----------------------------------------------
    # Generate agriculture advice
    # ----------------------------------------------

    with st.spinner(
        "Preparing agriculture advice..."
    ):

        result = adviser.answer(
            question=final_question,
            image_description=image_description
        )

    # ----------------------------------------------
    # Display answer
    # ----------------------------------------------

    st.markdown("### 🌱 Agriculture Adviser")

    st.markdown(
        f"""
        <div class="answer-box">
        {result["answer"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------
    # Display sources
    # ----------------------------------------------

    sources = result.get(
        "sources",
        []
    )

    if sources:

        st.markdown("### 📚 Agriculture Sources")

        for source in sources:

            st.write(
                f"• {source['source']} — "
                f"Page {source['page']}"
            )