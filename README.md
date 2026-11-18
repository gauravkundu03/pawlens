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

## Optional Google sign-in and account creation

The app uses Streamlit's OpenID Connect integration and does not store passwords. Users can sign in with Google or create a Google account through Google's account flow. The classifier remains available in guest mode when Google sign-in is not configured.

1. Create a new OAuth client in Google Cloud Console. Add `http://localhost:8501/oauth2callback` as an authorized redirect URI. If the consent screen is in testing mode, add your Google account as a test user.
2. Open `.streamlit/secrets.toml` and enter the new client ID and client secret. This file is gitignored; do not commit or share it.
3. Generate a cookie secret in the activated project environment:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Paste the generated value into `cookie_secret` in `.streamlit/secrets.toml`.
4. Restart Streamlit. The sidebar will offer Google sign-in and account creation.

When hosting the app elsewhere, update the redirect URI in both Streamlit secrets and the Google OAuth client settings. Never commit `secrets.toml`.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository. `.gitignore` excludes the large training artifacts but keeps `models/final_model.keras`.
2. Create an app at share.streamlit.io and set the main file to `app/app.py`.
3. Select Python 3.11. If enabling Google sign-in, configure the same OIDC values in Streamlit Community Cloud secrets, using the deployed app URL as the OAuth redirect URI.

## Project structure

- `app/app.py`: Streamlit classifier, account controls, experiment insights, and session history
- `src/`: reusable preprocessing, prediction, training, and feature modules
- `notebooks/`: capstone notebook
- `models/`, `results/`, `figures/`: trained model, evaluation outputs, and plots
- `requirements-train.txt`: additional packages for re-running training
