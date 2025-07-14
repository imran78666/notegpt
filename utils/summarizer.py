import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from groq import Groq
from utils.points_manager import add_points  # ✅ Import points manager

# ✅ Load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# ✅ Create Groq client
client = Groq(api_key=groq_api_key)


def get_summary(text):
    # Split text into smaller, safe chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    summaries = []

    # Summarize each chunk separately (helps prevent cutoff)
    for chunk in chunks[:6]:  # Limit max 6 chunks for long docs
        prompt = f"""
You are a helpful AI assistant. Summarize the following academic content in a clear and detailed way for study purposes.

CONTENT:
{chunk}

SUMMARY:
"""
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        summaries.append(response.choices[0].message.content.strip())

    # Combine small summaries into one
    combined_summary = "\n\n".join(summaries)

    # Ask AI to refine and clean up the final summary
    final_prompt = f"""
You are a helpful AI. Given the following partial summaries, combine and rewrite them into a clean, structured, and detailed academic summary.

PARTIAL SUMMARIES:
{combined_summary}

FINAL SUMMARY:
"""
    final_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": final_prompt.strip()}]
    )

    final_summary = final_response.choices[0].message.content.strip()

    # ✅ Add 2 points after successful summary
    if 'username' in st.session_state:
        add_points(st.session_state.username, 2)

    return final_summary
