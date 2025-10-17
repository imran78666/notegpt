import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from groq import Groq
from utils.points_manager import add_points  # ✅ Import points manager

# ✅ Load API key safely from secrets or environment
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key or not isinstance(groq_api_key, str):
    st.error("🚨 Missing or invalid GROQ_API_KEY. Please set it in Streamlit Secrets.")
    st.stop()

# ✅ Create Groq client safely
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()


def get_summary(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    summaries = []

    # Summarize each chunk separately (limit to 6 for long docs)
    for chunk in chunks[:6]:
        prompt = f"""
You are a helpful AI assistant. Summarize the following academic content clearly and in detail for study purposes.

CONTENT:
{chunk}

SUMMARY:
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt.strip()}],
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"Error summarizing chunk: {e}")
            continue

    combined_summary = "\n\n".join(summaries)

    # Refine the final combined summary
    final_prompt = f"""
You are a helpful AI. Combine and rewrite the following partial summaries into one clean, structured, and academic summary.

PARTIAL SUMMARIES:
{combined_summary}

FINAL SUMMARY:
"""
    try:
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": final_prompt.strip()}],
        )
        final_summary = final_response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating final summary: {e}")
        final_summary = combined_summary

    # ✅ Add points after success
    if "username" in st.session_state:
        add_points(st.session_state.username, 2)

    return final_summary
