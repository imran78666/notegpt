# 🎓 NoteGPT - AI Education Chatbot

An interactive AI-powered learning assistant that helps users study by uploading PDFs, generating quizzes, tracking their performance, and gaining experience points — all through a fun gamified interface.

---

## 🔧 Features

- 📄 **Upload PDFs** and auto-extract study content
- 🤖 **AI-Generated Quizzes** from any subject matter
- 💯 **Quiz Score Calculation** with time and attempts tracking
- 🎯 **Quiz Performance Prediction** using ML (pass/fail)
- 🏆 **Gamified Leaderboard System** with XP, levels, and points
- 🔐 **Secure Login/Signup** using hashed passwords
- 📊 Optional **Admin Dashboard** for progress tracking (coming soon)

---

## 🚀 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **AI**: [Groq API](https://groq.com/) (LLM for quiz/hint generation)
- **ML**: Scikit-learn (Random Forest for performance prediction)
- **Storage**: `CSV` for users, `JSON` for points
- **Auth**: `bcrypt` for password hashing
- **PDF Parsing**: PyMuPDF / pdfplumber

---

## 📂 Folder Structure

notegpt-app/
│
├── app.py # Main Streamlit app
├── hash_gen.py # Password hashing tool
├── users.csv # User credentials (hashed)
├── points.json # XP & level tracking
├── quiz_data.csv # Dataset for quiz performance prediction
│
├── utils/ # Core backend logic
│ ├── pdf_parser.py # PDF to text extraction
│ ├── summarizer.py # AI summarizer (LLM-based)
│ ├── qa_engine.py # Chunk-based Q&A logic
│ ├── quiz_generator.py # AI quiz & hints generator
│ ├── points_manager.py # XP & leaderboard logic
│ └── ml_model.py # ML training & prediction logic
│
├── pages/ # Streamlit multi-page UI
│ ├── 1_Quiz.py # Main quiz interface
│ ├── 2_Leaderboard.py # Live leaderboard
│ ├── 3_About.py # Info/about section
│ └── 4_PredictPerformance.py# ML-based performance predictor
│
└── requirements.txt # Python dependencies


---

## 🧪 Setup & Usage
### 1. Clone the Repository
git clone https://github.com/yourusername/ai-edu-chatbot.git
cd ai-edu-chatbot/notegpt-app

### 2. Install Dependencies

pip install -r ../requirements.txt
Ensure Python 3.8+ is installed.

### 3. Configure Your Groq API Key
export GROQ_API_KEY="your_groq_api_key"   # Linux/macOS
# or
set GROQ_API_KEY="your_groq_api_key"      # Windows


### 4. Train ML Model (once)
python -c "from utils.ml_model import train_model; train_model()"

### 5. Run the App 
streamlit run app.py (or) python -m streamlit run app.py
Visit: http://localhost:8501

### 6.📈 ML-Based Quiz Performance Predictor
Dataset: quiz_data.csv
Model: RandomForestClassifier
Input: quiz score %, time taken (in sec), number of attempts
Output: Pass / Fail prediction

UI: Available in 4_PredictPerformance.py (Streamlit page)

### 👥 Authentication
New users can sign up and are stored in users.csv
Passwords are hashed with bcrypt
XP and quiz performance tracked in points.json

### 📌 Notes
Works with any subject: B.Tech, Commerce, Medical, etc.
Quizzes are generated from uploaded PDFs, not from hardcoded topics.
Scoring and performance prediction help students understand learning gaps.


### 🧑‍💻 Author
Built by Shaik Mohammed Imran as a B.Tech final-year project fom Pragati Engineering College

Contact:munnaimran50@gmail.com

💬 Feel free to fork or contribute!



