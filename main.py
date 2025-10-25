import streamlit as st
from groq import Groq
import os
import re

# Set up your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

st.set_page_config(page_title="AI Interview System", layout="centered")
st.title("🤖 AI Interview System")
st.write("Generate structured interview questions and get AI feedback on answers.")

# Inputs
candidate_name = st.text_input("Candidate Name:")
job_role = st.text_input("Job Role (e.g., Software Engineer, Data Analyst):")
resume_text = st.text_area("Paste candidate's resume or skills summary:")
num_questions = st.slider("Number of questions:", 3, 10, 5)

# Initialize session state
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []

# Step 1: Generate Questions
if st.button("Generate Interview Questions"):
    if not client:
        st.error("Groq API key not found. Please set GROQ_API_KEY.")
    elif resume_text and job_role:
        with st.spinner("Generating questions..."):
            prompt = f"""
            You are an expert interviewer for the role of {job_role}.
            Candidate's skills and experience: {resume_text}

            Generate {num_questions} clear and specific interview questions (numbered 1 to {num_questions}).
            Avoid explanations or extra commentary—just the questions.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a professional interviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=700
                )

                text = response.choices[0].message.content.strip()
                # Extract only numbered questions
                st.session_state.questions = re.findall(r'\d+\.\s+(.*)', text)
                st.session_state.answers = [""] * len(st.session_state.questions)
                st.success("✅ Questions generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Enter job role and resume to generate questions.")

# Step 2: Answer Questions
if st.session_state.questions:
    st.subheader("📝 Answer the Questions")
    for i, q in enumerate(st.session_state.questions):
        st.session_state.answers[i] = st.text_area(f"{i+1}. {q}", value=st.session_state.answers[i])

    # Step 3: Analyze Answers
    if st.button("Analyze Answers"):
        with st.spinner("Analyzing answers..."):
            answer_prompt = "Analyze the following answers and give feedback, scores, and suggestions:\n\n"
            for i, q in enumerate(st.session_state.questions):
                answer_prompt += f"Q{i+1}: {q}\nAnswer: {st.session_state.answers[i]}\n\n"

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert HR and technical interviewer."},
                        {"role": "user", "content": answer_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )

                feedback = response.choices[0].message.content.strip()
                st.subheader("📊 Feedback & Analysis")
                st.write(feedback)
            except Exception as e:
                st.error(f"Error: {str(e)}")
