# MongoDB helpers for PawLens. Every function fails safely: if the database is
# not configured or unreachable, the app keeps working without saving anything.
from datetime import datetime, timezone

import streamlit as st
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError


@st.cache_resource(show_spinner=False)
def _client(uri):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")          # fail early if Atlas is unreachable
    return client


def get_db():
    # Returns the database object, or None when MongoDB is not configured/unreachable
    try:
        cfg = st.secrets["mongo"]
        uri = cfg["uri"]
        name = cfg.get("db", "pawlens")
    except Exception:
        return None
    if st.session_state.get("_db_failed"):
        return None
    try:
        db = _client(uri)[name]
        return db
    except Exception as error:            # bad password, IP not allowed, DNS, ...
        st.session_state["_db_failed"] = str(error)
        return None


def db_error():
    return st.session_state.get("_db_failed")


def _now():
    return datetime.now(timezone.utc)


def record_login(email, name):
    db = get_db()
    if db is None or not email:
        return False
    try:
        db.users.create_index("email", unique=True)
        db.users.update_one(
            {"email": email.lower()},
            {
                "$set": {"name": name or "", "last_login": _now()},
                "$setOnInsert": {"first_login": _now()},
                "$inc": {"login_count": 1},
            },
            upsert=True,
        )
        return True
    except PyMongoError:
        return False


def save_prediction(email, record):
    # Saves only prediction metadata (never the image itself)
    db = get_db()
    if db is None or not email:
        return False
    try:
        db.predictions.create_index([("email", 1), ("created_at", -1)])
        db.predictions.insert_one({
            "email": email.lower(),
            "file": record.get("file"),
            "prediction": record.get("prediction"),
            "prob_dog": record.get("prob_dog"),
            "prob_cat": record.get("prob_cat"),
            "confidence": record.get("confidence"),
            "created_at": _now(),
        })
        return True
    except PyMongoError:
        return False


def load_predictions(email, limit=200):
    db = get_db()
    if db is None or not email:
        return None
    try:
        cur = db.predictions.find({"email": email.lower()}, {"_id": 0, "email": 0}) \
                            .sort("created_at", DESCENDING).limit(limit)
        return list(cur)
    except PyMongoError:
        return None


def delete_predictions(email):
    db = get_db()
    if db is None or not email:
        return 0
    try:
        return db.predictions.delete_many({"email": email.lower()}).deleted_count
    except PyMongoError:
        return 0


def list_users():
    db = get_db()
    if db is None:
        return None
    try:
        return list(db.users.find({}, {"_id": 0}).sort("last_login", DESCENDING))
    except PyMongoError:
        return None


def prediction_stats():
    db = get_db()
    if db is None:
        return None
    try:
        total = db.predictions.count_documents({})
        by_label = {d["_id"]: d["n"] for d in db.predictions.aggregate(
            [{"$group": {"_id": "$prediction", "n": {"$sum": 1}}}])}
        return {"total": total, "by_label": by_label}
    except PyMongoError:
        return None
