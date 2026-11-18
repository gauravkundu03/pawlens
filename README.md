# PawLens: Cat vs Dog Image Classification

The final model uses fine-tuned MobileNetV2 transfer learning. Its recorded evaluation is **98.4% test accuracy** and **0.999 ROC-AUC** (see `results/final_model_comparison.csv`).

## Run locally

```powershell
cd C:\Users\gaura\Downloads\Cat-Dog-Classification
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app\app.py
```

Open http://localhost:8501 to classify an image, explore model metrics, and download the current session's prediction history. On macOS/Linux, activate the environment with `source .venv/bin/activate` and run `python -m streamlit run app/app.py`.

## Required Google sign-in and account creation

The classifier requires Google sign-in. The app uses Streamlit's OpenID Connect integration and does not store passwords. New users can create a Google account through Google's account flow. If authentication is not configured, the app displays a setup message and does not expose the classifier.

1. Create a Web application OAuth client in Google Cloud Console. Add `http://localhost:8501/oauth2callback` as an authorized redirect URI. If the consent screen is in testing mode, add your Google account as a test user.
2. Open `.streamlit/secrets.toml` and enter the new client ID and client secret. This file is gitignored; do not commit or share it.
3. Generate a cookie secret in the activated project environment:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Paste the generated value into `cookie_secret` in `.streamlit/secrets.toml`.
4. Restart Streamlit. Sign in with Google to access the classifier. The Google account flow also lets new users create an account.

When hosting the app elsewhere, update the redirect URI in both Streamlit secrets and the Google OAuth client settings. Never commit `secrets.toml`.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository. `.gitignore` excludes the local `.streamlit/secrets.toml` and large training artifacts but keeps `models/final_model.keras`.
2. Create an app at share.streamlit.io and set the main file to `app/app.py`.
3. Select Python 3.11.
4. In the app's Settings → Secrets, add the same `[auth]` and `[auth.google]` values from `.streamlit/secrets.toml`, but set `redirect_uri` to `https://YOUR-APP-NAME.streamlit.app/oauth2callback`.
5. Add that exact deployed redirect URI to the Google OAuth client's authorized redirect URIs, then reboot the app.

## Project structure

- `app/app.py`: Streamlit classifier, account controls, experiment insights, and session history
- `src/`: reusable preprocessing, prediction, training, and feature modules
- `notebooks/`: capstone notebook
- `models/`, `results/`, `figures/`: trained model, evaluation outputs, and plots
- `requirements-train.txt`: additional packages for re-running training
