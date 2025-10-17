import streamlit as st
from streamlit import session_state as st_session
from groq import Groq
import nltk
import random
import json
import re
from nltk import sent_tokenize
from utils.points_manager import add_points

# -------------------------------
# ✅ Ensure NLTK punkt is available
# -------------------------------
try:
    sent_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)

# -------------------------------
# ✅ Initialize Groq client
# -------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    raise KeyError(
        'GROQ_API_KEY not found in Streamlit secrets. '
        'Add it via Manage App → Secrets.'
    )

client = Groq(api_key=groq_api_key)

# -------------------------------
# ✅ Helper functions
# -------------------------------

def is_heading(line: str) -> bool:
    words = line.strip().split()
    return len(words) <= 8 and line.isupper()


def split_sentences(sentences: list[str], max_chars: int = 1500) -> list[str]:
    chunks = []
    current_chunk = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(s)
        current_len += len(s)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def clean_hint(hint: str) -> str:
    if not hint.strip():
        return "Think carefully about the concept."
    bad_keywords = ["check the text", "read carefully", "look at", "refer"]
    if any(bad_word in hint.lower() for bad_word in bad_keywords):
        return "Review the concept logically."
    return hint


def extract_json_array(text: str) -> list[dict]:
    """Safely extract JSON array from Groq's output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback using regex
        match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                print("Parsing failed:", e)
        return []

# -------------------------------
# ✅ Main quiz generator functions
# -------------------------------

def generate_quiz(text: str, count: int = 5) -> list[tuple]:
    sentences = [s for s in sent_tokenize(text) if not is_heading(s)]
    if not sentences:
        return []

    random.shuffle(sentences)
    chunks = split_sentences(sentences)
    quizzes = []

    for chunk in chunks:
        if len(quizzes) >= count:
            break

        prompt = f"""
You are a professional educational assistant. Create 3 multiple-choice questions (MCQs) from the following academic content.
Each question must have exactly 4 options (A, B, C, D). Provide a helpful hint and clear explanation.

Return strictly this JSON format:
[
  {{
    "question": "...",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A",
    "hint": "...",
    "explanation": "..."
  }},
  ...
]
TEXT:
{chunk}

Return ONLY the JSON list. No extra text.
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            items = extract_json_array(content)

            for item in items:
                options = item.get("options", [])
                if len(options) == 4 and item.get("answer") in options:
                    quizzes.append((
                        item.get("question", ""),
                        options,
                        item.get("answer", ""),
                        clean_hint(item.get("hint", "")),
                        item.get("explanation", "")
                    ))
        except Exception as e:
            print("Quiz generation failed:", e)
            continue

    if 'username' in st_session:
        add_points(st_session.username, 3)

    return quizzes[:count]


def generate_quiz_by_topic(text: str, topic: str, num: int) -> list[tuple]:
    sentences = [s for s in sent_tokenize(text) if not is_heading(s)]
    if not sentences:
        return []

    random.shuffle(sentences)
    chunks = split_sentences(sentences)
    quizzes = []

    for chunk in chunks:
        if len(quizzes) >= num:
            break

        prompt = f"""
You are a professional educational assistant. Create 3 multiple-choice questions (MCQs) focusing on the topic '{topic}' from this content.
Each question must have exactly 4 options (A, B, C, D). Provide a helpful hint and clear explanation.

Return strictly this JSON format:
[
  {{
    "question": "...",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A",
    "hint": "...",
    "explanation": "..."
  }},
  ...
]
TEXT:
{chunk}

Return ONLY the JSON list. No extra text.
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            items = extract_json_array(content)

            for item in items:
                options = item.get("options", [])
                if len(options) == 4 and item.get("answer") in options:
                    quizzes.append((
                        item.get("question", ""),
                        options,
                        item.get("answer", ""),
                        clean_hint(item.get("hint", "")),
                        item.get("explanation", "")
                    ))
        except Exception as e:
            print("Quiz generation failed:", e)
            continue

    return quizzes[:num]
