"""
db_utils.py  –  Supabase backend for MindGuard
Reads credentials from .streamlit/secrets.toml via st.secrets:
    [supabase]
    url = "https://xxxx.supabase.co"
    key = "your-anon-or-service-role-key"

Required Supabase tables (run once in the SQL editor):
──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,           -- bcrypt hash
    is_admin   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assessments (
    id             BIGSERIAL PRIMARY KEY,
    user_id        BIGINT REFERENCES users(id) ON DELETE CASCADE,
    username       TEXT NOT NULL,
    timestamp      TIMESTAMPTZ DEFAULT NOW(),
    predicted_risk TEXT NOT NULL,
    probabilities  TEXT NOT NULL,       -- JSON array string
    inputs         TEXT NOT NULL        -- JSON object string
);
──────────────────────────────────────────────────────
"""

import json
import hashlib
import streamlit as st
from supabase import create_client, Client


# ── Supabase client (cached) ─────────────────────────────────────────────────
@st.cache_resource
def _get_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


# ── Password helpers ─────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """SHA-256 hash (simple, no extra deps).
       For production, swap for bcrypt: pip install bcrypt."""
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(password: str, hashed: str) -> bool:
    return _hash_password(password) == hashed


# ── User management ──────────────────────────────────────────────────────────
def register_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (True, '') on success or (False, error_message) on failure."""
    try:
        client = _get_client()
        # Check if username already exists
        existing = (
            client.table("users")
            .select("id")
            .eq("username", username)
            .execute()
        )
        if existing.data:
            return False, "Username already taken."

        hashed = _hash_password(password)
        client.table("users").insert(
            {"username": username, "password": hashed, "is_admin": False}
        ).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def verify_login(username: str, password: str) -> tuple[bool, int | None, bool]:
    """Returns (success, user_id, is_admin)."""
    try:
        client = _get_client()
        result = (
            client.table("users")
            .select("id, password, is_admin")
            .eq("username", username)
            .execute()
        )
        if not result.data:
            return False, None, False
        row = result.data[0]
        if _verify_password(password, row["password"]):
            return True, row["id"], row["is_admin"]
        return False, None, False
    except Exception:
        return False, None, False


# ── Assessment management ────────────────────────────────────────────────────
def save_assessment(
    user_id: int,
    predicted_risk: str,
    probabilities: list,
    inputs_dict: dict,
) -> None:
    """Save a prediction result for a user."""
    try:
        client = _get_client()

        # Resolve username from user_id
        user_row = (
            client.table("users")
            .select("username")
            .eq("id", user_id)
            .execute()
        )
        username = user_row.data[0]["username"] if user_row.data else "unknown"

        client.table("assessments").insert(
            {
                "user_id": user_id,
                "username": username,
                "predicted_risk": predicted_risk,
                "probabilities": json.dumps(probabilities),
                "inputs": json.dumps(inputs_dict),
            }
        ).execute()
    except Exception as e:
        st.warning(f"Could not save assessment: {e}")


def get_all_assessments() -> list:
    """
    Returns a list of tuples matching the original SQLite schema:
        (id, username, timestamp, predicted_risk, probabilities, inputs)
    """
    try:
        client = _get_client()
        result = (
            client.table("assessments")
            .select("id, username, timestamp, predicted_risk, probabilities, inputs")
            .order("timestamp", desc=True)
            .execute()
        )
        rows = []
        for row in result.data:
            rows.append((
                row["id"],
                row["username"],
                row["timestamp"],
                row["predicted_risk"],
                row["probabilities"],
                row["inputs"],
            ))
        return rows
    except Exception:
        return []


def get_user_assessments(user_id: int) -> list:
    """
    Returns assessments for a single user as a list of tuples:
        (id, username, timestamp, predicted_risk, probabilities, inputs)
    """
    try:
        client = _get_client()
        result = (
            client.table("assessments")
            .select("id, username, timestamp, predicted_risk, probabilities, inputs")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .execute()
        )
        rows = []
        for row in result.data:
            rows.append((
                row["id"],
                row["username"],
                row["timestamp"],
                row["predicted_risk"],
                row["probabilities"],
                row["inputs"],
            ))
        return rows
    except Exception:
        return []


def delete_assessment(assessment_id: int) -> None:
    """Delete a single assessment by ID."""
    try:
        client = _get_client()
        client.table("assessments").delete().eq("id", assessment_id).execute()
    except Exception as e:
        st.warning(f"Could not delete assessment: {e}")