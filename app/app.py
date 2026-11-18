import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import load_trained_model, predict_image
from preprocessing import InvalidImageError

MODEL_PATH = ROOT / "models" / "final_model.keras"
COMPARISON_PATH = ROOT / "results" / "final_model_comparison.csv"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"

st.set_page_config(
    page_title="PawLens | Cat and dog classifier",
    page_icon=":material/pets:",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_model_card():
    with MODEL_CARD_PATH.open(encoding="utf-8") as file:
        return json.load(file)


if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

try:
    auth_settings = st.secrets.get("auth", {})
except StreamlitSecretNotFoundError:
    auth_settings = {}
google_settings = auth_settings.get("google", {})
google_auth_configured = all(
    (
        auth_settings.get("redirect_uri"),
        auth_settings.get("cookie_secret"),
        google_settings.get("client_id"),
        google_settings.get("client_secret"),
        google_settings.get("server_metadata_url"),
    )
)
is_logged_in = google_auth_configured and st.user.is_logged_in

st.title("PawLens", icon=":material/pets:")
st.markdown("### A closer look at your furry friends")
st.caption(
    "An interactive cat and dog image classifier powered by a fine-tuned MobileNetV2 model."
)
st.badge("MobileNetV2", icon=":material/neurology:")
st.badge("Cat and dog", icon=":material/pets:", color="orange")

with st.sidebar:
    st.header("Your account", icon=":material/account_circle:")
    if is_logged_in:
        st.success(f"Signed in as {st.user.name or st.user.email}")
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            st.logout()
    else:
        account_mode = st.segmented_control(
            "Account",
            options=["Sign in", "Create account"],
            default="Sign in",
            key="account_mode",
            width="stretch",
        )
        if google_auth_configured:
            button_label = (
                "Continue with Google"
                if account_mode == "Sign in"
                else "Create account with Google"
            )
            if st.button(
                button_label,
                type="primary",
                icon=":material/login:",
                width="stretch",
            ):
                st.login("google")
            if account_mode == "Create account":
                st.caption(
                    "Google will guide you through signing in or creating a Google account."
                )
        else:
            st.button(
                "Google sign-in is not configured",
                disabled=True,
                width="stretch",
            )
            st.caption("You can still explore the classifier as a guest.")
            with st.expander("Set up Google sign-in"):
                st.write(
                    "Follow the Google OAuth setup steps in the project README, "
                    "then add your credentials to `.streamlit/secrets.toml`."
                )
    st.caption("Uploaded images are analyzed by the local model.")

classifier_tab, insights_tab, history_tab = st.tabs(
    ["Classifier", "Model insights", "History"]
)

with classifier_tab:
    st.subheader("Try an image", icon=":material/center_focus_strong:")
    st.write(
        "Upload a cat or dog photo, choose how confident the model should be, "
        "and run the prediction."
    )

    with st.form("classifier_form"):
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, PNG, BMP, and WebP.",
        )
        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.50,
            max_value=0.95,
            value=0.60,
            step=0.01,
            help="Predictions below this threshold are marked Uncertain.",
        )
        submitted = st.form_submit_button(
            "Analyze image",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if submitted:
        if uploaded_image is None:
            st.warning("Choose an image before starting the analysis.")
        else:
            try:
                with st.spinner("Loading the model and analyzing your image..."):
                    model = get_model()
                    prediction = predict_image(model, uploaded_image)
                is_uncertain = prediction["confidence"] < min_confidence
                label = "Uncertain" if is_uncertain else prediction["label"]
                prob_dog = 0.0 if is_uncertain else prediction["prob_dog"]
                prob_cat = 0.0 if is_uncertain else prediction["prob_cat"]
                confidence = 0.0 if is_uncertain else prediction["confidence"]
                result = {
                    "file": uploaded_image.name,
                    "image": prediction["image"],
                    "label": label,
                    "prob_dog": prob_dog,
                    "prob_cat": prob_cat,
                    "confidence": confidence,
                }
                st.session_state.last_result = result
                st.session_state.history.append(
                    {
                        "file": result["file"],
                        "prediction": label,
                        "prob_dog": round(prob_dog, 4),
                        "prob_cat": round(prob_cat, 4),
                        "confidence": round(confidence, 4),
                    }
                )
                if is_uncertain:
                    st.toast(
                        "The model is unsure. Try a clearer photo.",
                        icon=":material/help:",
                    )
                else:
                    st.toast("Analysis complete!", icon=":material/check_circle:")
                    st.balloons()
            except InvalidImageError as error:
                st.error(f"That image could not be read. Please choose another one. {error}")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    result = st.session_state.last_result
    if result is not None:
        st.subheader(f"Prediction: {result['label']}")
        image_column, results_column = st.columns([1, 1.5], vertical_alignment="center")
        image_column.image(
            Image.fromarray(result["image"]),
            caption=result["file"],
            alt="Image analyzed by the cat and dog classifier",
        )
        with results_column:
            dog_column, cat_column, confidence_column = st.columns(3)
            dog_column.metric("P(Dog)", f"{result['prob_dog']:.3f}", border=True)
            cat_column.metric("P(Cat)", f"{result['prob_cat']:.3f}", border=True)
            confidence_column.metric(
                "Confidence", f"{result['confidence'] * 100:.1f}%", border=True
            )
            st.progress(
                float(result["prob_dog"]),
                text="Probability of Dog",
            )
            st.caption(
                "Uncertain results show zero probability and confidence by design."
            )
    else:
        st.info(
            "Your result will appear here after you analyze an image.",
            icon=":material/info:",
        )

with insights_tab:
    st.subheader("Model insights", icon=":material/analytics:")
    try:
        model_card = load_model_card()
        comparison = load_model_comparison()
        final_model = comparison.loc[
            comparison["Approach"] == "Transfer Learning"
        ].iloc[0]
        accuracy_column, auc_column, model_column = st.columns(3)
        accuracy_column.metric(
            "Test accuracy", f"{final_model['Accuracy'] * 100:.1f}%", border=True
        )
        auc_column.metric("ROC-AUC", f"{final_model['ROC_AUC']:.3f}", border=True)
        model_column.metric("Model", model_card["final_model"], border=True)

        st.markdown("#### Compare experiments")
        metric_options = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        selected_metric = st.selectbox("Rank models by", metric_options)
        leaderboard = comparison.sort_values(selected_metric, ascending=False)
        st.dataframe(
            leaderboard[["Approach", selected_metric]].reset_index(drop=True),
            hide_index=True,
        )
        st.caption(
            f"Model input: {model_card['img_size']} × {model_card['img_size']} RGB. "
            f"Classes: {', '.join(model_card['classes'])}."
        )
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load the saved model information: {error}")

with history_tab:
    st.subheader("Your session history", icon=":material/history:")
    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, hide_index=True)
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="pawlens_prediction_history.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        if st.button("Clear history", icon=":material/delete:"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info(
            "Predictions from this browser session will show up here.",
            icon=":material/history:",
        )
