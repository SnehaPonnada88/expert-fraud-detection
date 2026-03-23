import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.env import ExpertEnv
from src.feature_engineering import (
    compute_depth_from_llm,
    compute_consistency_from_llm,
    compute_progression,
    compute_expertise_score,
    compute_profile_anomaly   # ✅ NEW
)
from src.llm_extractor import extract_signals

st.set_page_config(page_title="Expert Fraud Detection", layout="wide")

st.title("Expert Fraud Detection System")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "initial"

if "llm_outputs" not in st.session_state:
    st.session_state.llm_outputs = []

if "depth_scores" not in st.session_state:
    st.session_state.depth_scores = []

if "followup_question" not in st.session_state:
    st.session_state.followup_question = ""

if "initial_answers" not in st.session_state:
    st.session_state.initial_answers = []

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
df = pd.read_csv("data/synthetic.csv")
model = PPO.load("models/ppo_model")
env = ExpertEnv(df)

# -----------------------------
# SELECT CANDIDATE
# -----------------------------
idx = st.slider("Select Candidate", 0, len(df)-1, 0)
row = df.iloc[idx]

profile = ast.literal_eval(row["profile_history"])

# ✅ NEW: compute profile anomaly
profile_score = compute_profile_anomaly(profile)

latest_skills = profile[-1]["skills"]
selected_skill = latest_skills[0] if latest_skills else "technology"

# -----------------------------
# PROFILE VISUALIZATION
# -----------------------------
skill_counts = [len(p["skills"]) for p in profile]
timestamps = [p["timestamp"] for p in profile]

st.subheader("Skill Growth Over Time")
fig, ax = plt.subplots()
ax.plot(timestamps, skill_counts, marker='o')
ax.set_ylabel("Skill Count")
st.pyplot(fig)

# ✅ NEW: show anomaly below graph
st.caption(f"Profile Anomaly Score: {round(profile_score, 3)}")

# -----------------------------
# SCREENING QUESTIONS
# -----------------------------
st.subheader("Screening Questions")
st.write(f"Focus Area: **{selected_skill}**")

q1 = st.text_area(f"Q1: Describe a project where you used {selected_skill}")
q2 = st.text_area("Q2: What went wrong in that project?")
q3 = st.text_area("Q3: How did you debug it?")

answers = [q1, q2, q3]

# -----------------------------
# START EVALUATION
# -----------------------------
if st.button("Evaluate Candidate"):
    st.session_state.stage = "evaluated"
    st.session_state.llm_outputs = []
    st.session_state.depth_scores = []
    st.session_state.initial_answers = answers
    st.rerun()

# -----------------------------
# STAGE 1: INITIAL EVALUATION
# -----------------------------
if st.session_state.stage == "evaluated":

    valid_answers = [a for a in st.session_state.initial_answers if a.strip()]

    if not valid_answers:
        st.warning("Please answer at least one question.")
        st.stop()

    st.subheader("LLM Extracted Signals")

    for i, ans in enumerate(valid_answers):
        out = extract_signals(ans)
        st.session_state.llm_outputs.append(out)

        st.write(f"Answer {i+1}:", out)

        score = out.get("specificity", 0.3)
        if out.get("has_reasoning"):
            score += 0.2
        if out.get("has_failure"):
            score += 0.2

        st.session_state.depth_scores.append(min(score, 1.0))

    # -----------------------------
    # COMPUTE FEATURES
    # -----------------------------
    depth = compute_depth_from_llm(st.session_state.llm_outputs)
    consistency = compute_consistency_from_llm(st.session_state.llm_outputs, depth)
    progression = compute_progression(st.session_state.depth_scores, depth)

    expertise_score = compute_expertise_score(depth, consistency, progression)

    st.subheader("Computed Features")
    st.write({
        "Profile Anomaly": round(profile_score, 3),   # ✅ NEW
        "Depth": round(depth, 3),
        "Consistency": round(consistency, 3),
        "Progression": round(progression, 3),
        "Expertise Score": round(expertise_score, 3)
    })

    # ✅ NEW: adjusted expertise (light profile impact)
    adjusted_expertise = expertise_score * (1 - 0.3 * profile_score)
    fraud_probability = round(1 - adjusted_expertise, 3)

    st.subheader("Fraud Risk Score")
    st.progress(fraud_probability)

    # -----------------------------
    # PROBE LOGIC
    # -----------------------------
    if 0.35 <= expertise_score <= 0.7:
        st.session_state.stage = "probe"

        weakest = min(
            {"depth": depth, "consistency": consistency, "progression": progression},
            key=lambda x: {"depth": depth, "consistency": consistency, "progression": progression}[x]
        )

        if weakest == "depth":
            st.session_state.followup_question = "Can you provide more technical implementation details?"
        elif weakest == "consistency":
            st.session_state.followup_question = "Are all your answers referring to the same project?"
        else:
            st.session_state.followup_question = "Can you explain your debugging process step-by-step?"

        st.rerun()

    else:
        st.session_state.stage = "final"
        st.rerun()

# -----------------------------
# STAGE 2: FOLLOW-UP
# -----------------------------
if st.session_state.stage == "probe":

    st.subheader("Follow-up Question")
    follow_up = st.text_area(st.session_state.followup_question)

    if st.button("Submit Follow-up Answer"):

        extra_output = extract_signals(follow_up)
        st.write("Follow-up Analysis:", extra_output)

        st.session_state.llm_outputs.append(extra_output)

        score = extra_output.get("specificity", 0.3)
        if extra_output.get("has_reasoning"):
            score += 0.2
        if extra_output.get("has_failure"):
            score += 0.2

        st.session_state.depth_scores.append(min(score, 1.0))

        st.session_state.stage = "final"
        st.rerun()

# -----------------------------
# STAGE 3: FINAL DECISION
# -----------------------------
if st.session_state.stage == "final":

    depth = compute_depth_from_llm(st.session_state.llm_outputs)
    consistency = compute_consistency_from_llm(st.session_state.llm_outputs, depth)
    progression = compute_progression(st.session_state.depth_scores, depth)

    expertise_score = compute_expertise_score(depth, consistency, progression)

    # ✅ SAME adjusted logic here
    adjusted_expertise = expertise_score * (1 - 0.3 * profile_score)
    fraud_probability = round(1 - adjusted_expertise, 3)

    st.subheader("Final Decision")

    if adjusted_expertise > 0.75:
        st.success("✅ Strong Candidate (Low Fraud Risk)")
    elif adjusted_expertise > 0.5:
        st.warning("⚠️ Needs Further Evaluation")
    else:
        st.error("🚨 High Risk of Overstated Expertise")

    st.write({
        "Fraud Probability": fraud_probability,
        "Expertise Score": round(adjusted_expertise, 3)
    })

    st.subheader("Why this decision?")

    reasons = []

    if depth < 0.4:
        reasons.append("Answers lack technical depth")

    if consistency < 0.5:
        reasons.append("Weak consistency across responses")

    if progression < 0.5:
        reasons.append("No meaningful progression")

    if profile_score > 0.6:
        reasons.append("Unusual profile growth detected")  # ✅ NEW

    if adjusted_expertise > 0.7:
        reasons.append("Strong overall expertise signals")

    if not reasons:
        reasons.append("Moderate signals detected")

    st.write(reasons)