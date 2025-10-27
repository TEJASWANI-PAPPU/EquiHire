# app.py
import streamlit as st
from groq import Groq
import os
import re
import json
import uuid
import datetime
import pandas as pd
import numpy as np

# ---------- CONFIG & CLIENT ----------
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None

st.set_page_config(page_title="FairHire — AI Interview System", layout="wide")
st.title("🤖 FairHire — AI Interview System (Prototype)")

# ---------- HELPERS ----------
def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def extract_skills_from_text(text):
    """
    Lightweight skill extraction: returns a dict of skills with a confidence score.
    For demo/hackathon: keyword/regex approach. Replace with embeddings+ner in prod.
    """
    # simple seed skill list (extend for your domain)
    seed_skills = [
        "python","java","javascript","react","spring boot","django","sql","nosql",
        "aws","docker","kubernetes","data structures","algorithms","rest api",
        "machine learning","nlp","git","html","css","flask","spring","c++","c#"
    ]
    lowered = text.lower()
    found = {}
    for s in seed_skills:
        if s in lowered:
            # crude confidence: more occurrences -> higher confidence
            score = min(0.2 + lowered.count(s) * 0.15, 0.95)
            found[s] = round(score, 2)
    # also try to capture "years of experience" patterns
    yrs = re.findall(r'(\d+)\s+years?', text.lower())
    if yrs:
        found["years_reported"] = max(int(x) for x in yrs)
    return found

def build_skills_graph(skills_dict):
    """
    Build a very small skills graph structure for demo:
    nodes: skills (with confidence)
    edges: inferred relationships by simple co-occurrence rules
    """
    nodes = []
    edges = []
    for s,score in skills_dict.items():
        if s == "years_reported":
            continue
        nodes.append({"id": s, "score": score})
    # naive edges: languages -> frameworks
    map_edges = [
        ("python", "django"), ("python", "flask"),
        ("javascript", "react"), ("java", "spring boot"),
        ("sql", "nosql")
    ]
    for a,b in map_edges:
        if any(n["id"]==a for n in nodes) and any(n["id"]==b for n in nodes):
            edges.append({"source": a, "target": b, "weight": 0.8})
    return {"nodes": nodes, "edges": edges}

def append_audit(entry):
    """ Append audit entry to session state audit log (in-memory prototype) """
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    st.session_state.audit_log.append(entry)

def simulate_score_answer_with_human_rubric(answer_text, question_text):
    """
    For demo: a simple heuristic scorer. In prod this should be LLM-rubric based scoring
    or automated tests / code execution for code questions.
    """
    length = len(answer_text.strip())
    keywords = ["experience","project","describe","architecture","trade-off","complexity","optimize","test"]
    hits = sum(1 for k in keywords if k in answer_text.lower())
    score = min(10, max(0, (length/50) + hits))
    return round(score,1)

def compute_group_metrics(batch_df, score_column="score", group_col="group"):
    """
    Compute simple fairness metrics:
    - selection_rate (>= threshold) per group
    - adverse impact ratio: min selection rate / max selection rate
    """
    threshold = batch_df[score_column].mean()  # demo threshold
    sel = batch_df[score_column] >= threshold
    rates = batch_df.groupby(group_col)[sel.name if isinstance(sel, pd.Series) else slice(None)].apply(lambda g: (batch_df.loc[g.index, score_column] >= threshold).mean() if len(g)>0 else np.nan)
    # simpler: group selection rates
    group_rates = {}
    for g,grp in batch_df.groupby(group_col):
        group_rates[g] = float((grp[score_column] >= threshold).mean())
    min_rate = min(group_rates.values()) if group_rates else 0.0
    max_rate = max(group_rates.values()) if group_rates else 1.0
    adverse_impact = (min_rate / max_rate) if max_rate>0 else None
    return {"threshold": float(threshold), "group_rates": group_rates, "adverse_impact_ratio": adverse_impact}

# ---------- UI: Sidebar & Inputs ----------
with st.sidebar:
    st.header("Interview Configuration")
    job_role = st.text_input("Job Role (e.g., Software Engineer):", value="Software Engineer")
    candidate_name = st.text_input("Candidate Name:")
    resume_text = st.text_area("Paste resume or skills summary (plain text):", height=120)
    num_questions = st.slider("Number of questions:", 3, 10, 5)
    demo_batch = st.file_uploader("Upload batch CSV for bias demo (optional)", type=["csv"])
    st.markdown("---")
    st.write("⚠️ This is a prototype. Don't store PII here in prod.")
    if not client:
        st.warning("Groq API key not found — LLM functionality will be disabled in demo mode.")

# ---------- Step 0: Skills graph (local) ----------
if st.button("Build Skills Graph from Resume"):
    if not resume_text:
        st.warning("Provide resume text first.")
    else:
        skills = extract_skills_from_text(resume_text)
        graph = build_skills_graph(skills)
        st.session_state.skills = skills
        st.session_state.skills_graph = graph
        append_audit({
            "id": str(uuid.uuid4()),
            "timestamp": now_iso(),
            "event": "skills_graph_built",
            "candidate": candidate_name,
            "skills": skills
        })
        st.success("Skills graph created (prototype).")
        st.json(graph)

# ---------- Step 1: Generate Questions (LLM) ----------
if st.button("Generate Interview Questions"):
    if not client:
        st.error("Groq API key not configured. Questions can't be generated via LLM in demo mode.")
    elif not (resume_text and job_role):
        st.warning("Provide job role and resume text.")
    else:
        prompt = f"""
You are an expert interviewer for the role of {job_role}.
Candidate's skills and experience: {resume_text}
Use the skills graph concept and job description to generate {num_questions} numbered interview questions,
and for each question include a short 'rationale' line explaining why the question is relevant.
Return JSON array of objects: [{{"qnum":1,"question":"...","rationale":"..."}}, ...]
"""
        try:
            # call LLM
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role":"system","content":"You are a professional interviewer and output JSON only."},
                    {"role":"user","content": prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )
            text = response.choices[0].message.content.strip()
            # Try to parse JSON from response
            try:
                qlist = json.loads(text)
            except Exception:
                # fallback: extract numbered lines
                qlist = []
                lines = text.splitlines()
                for i,l in enumerate(lines):
                    m = re.match(r'\d+\.\s*(.*)', l)
                    if m:
                        q = m.group(1).strip()
                        qlist.append({"qnum": i+1, "question": q, "rationale": ""})
            st.session_state.questions = qlist
            append_audit({
                "id": str(uuid.uuid4()),
                "timestamp": now_iso(),
                "event": "questions_generated",
                "candidate": candidate_name,
                "questions_count": len(qlist),
                "job_role": job_role
            })
            st.success(f"Generated {len(qlist)} questions.")
        except Exception as e:
            st.error(f"LLM error: {e}")

# Show generated questions (if present)
if st.session_state.get("questions"):
    st.subheader("Generated Questions")
    for q in st.session_state.questions:
        st.write(f"**{q.get('qnum')}**. {q.get('question')}")
        if q.get("rationale"):
            st.caption(f"Rationale: {q.get('rationale')}")

# ---------- Step 2: Answer Questions and Scoring ----------
if st.session_state.get("questions"):
    st.subheader("Answer the Questions")
    if "candidate_answers" not in st.session_state:
        st.session_state.candidate_answers = {q["qnum"]: "" for q in st.session_state.questions}
    cols = st.columns(2)
    # render in two columns
    for q in st.session_state.questions:
        qnum = q["qnum"]
        with cols[0]:
            st.markdown(f"**Q{qnum}. {q['question']}**")
            ans = st.text_area(f"Answer Q{qnum}", value=st.session_state.candidate_answers[qnum], key=f"ans_{qnum}", height=120)
            st.session_state.candidate_answers[qnum] = ans
    if st.button("Auto-Score Answers (Demo heuristics)"):
        scores = {}
        for q in st.session_state.questions:
            qnum = q["qnum"]
            ans = st.session_state.candidate_answers[qnum]
            score = simulate_score_answer_with_human_rubric(ans, q["question"])
            scores[qnum] = score
        st.session_state.latest_scores = scores
        append_audit({
            "id": str(uuid.uuid4()),
            "timestamp": now_iso(),
            "event": "answers_scored",
            "candidate": candidate_name,
            "scores": scores
        })
        st.success("Answers scored (demo).")
    if st.session_state.get("latest_scores"):
        st.subheader("Scores")
        st.table(pd.DataFrame([
            {"Question": f"Q{qnum}", "Score": sc}
            for qnum, sc in st.session_state.latest_scores.items()
        ]))

# ---------- Fairness Dashboard (Prototype) ----------
st.header("Fairness & Bias Dashboard (Prototype)")
st.write("Upload a small CSV with columns: candidate_id, group (e.g., gender), score (numeric). We'll compute metrics.")

sample_df = None
if demo_batch is not None:
    try:
        batch_df = pd.read_csv(demo_batch)
        # basic validation
        if not set(["candidate_id","group","score"]).issubset(batch_df.columns):
            st.error("CSV must contain columns: candidate_id, group, score")
        else:
            st.success("Batch loaded.")
            st.dataframe(batch_df.head(20))
            metrics = compute_group_metrics(batch_df, score_column="score", group_col="group")
            st.metric("Threshold (mean score)", metrics["threshold"])
            st.write("Selection rates by group:")
            st.json(metrics["group_rates"])
            st.write("Adverse impact ratio (min_rate / max_rate):")
            st.write(metrics["adverse_impact_ratio"])
            # simple heatmap-like table
            gr = pd.DataFrame.from_dict(metrics["group_rates"], orient="index", columns=["selection_rate"])
            st.bar_chart(gr)
            append_audit({
                "id": str(uuid.uuid4()),
                "timestamp": now_iso(),
                "event": "batch_bias_analysis",
                "groups": list(metrics["group_rates"].keys()),
                "adverse_impact_ratio": metrics["adverse_impact_ratio"]
            })
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
else:
    st.info("Upload a demo CSV to compute group fairness metrics.")

# ---------- Audit Trail Viewer ----------
st.header("Audit Trail (Demo)")
if st.session_state.get("audit_log"):
    logs = st.session_state.audit_log
    df_logs = pd.DataFrame(logs)
    st.dataframe(df_logs[["timestamp","event","candidate"]].sort_values("timestamp", ascending=False))
    if st.button("Export audit log JSON"):
        fname = "/tmp/audit_log.json"
        with open(fname, "w") as f:
            json.dump(logs, f, indent=2)
        st.success("Saved to /tmp/audit_log.json (demo).")
else:
    st.write("No audit events yet. Trigger actions to record events.")

# ---------- Integration / Next Steps (Buttons demonstrating hooks) ----------
st.header("Integration & Next Steps")
st.write("""
This prototype demonstrates:
- skills graph extraction (local)
- LLM-driven question generation (via Groq) with rationale
- simple auto-scoring heuristics
- batch bias metrics (adverse impact)
- in-memory audit trail

Next steps for production:
- Replace simple skill extraction with NER + embeddings + Vector DB (FAISS/Pinecone)
- Store audit logs in tamper-evident storage (append-only ledger)
- Strong PII protection: encryption, redaction, and role-based access
- Ground LLM prompts with retrieved role-specific rubrics from DB
- Add multimodal answer evaluation: code execution sandbox, voice->transcript, video analysis
""")
