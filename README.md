# 🧠 Expert Fraud Detection System

Detecting overstated expertise through **consistency, depth, and behavior under probing**

---

## 🚀 Overview

In modern hiring, candidates can generate polished answers using AI tools.
This makes surface-level evaluation unreliable.

This project focuses on a deeper question:

> **Can a candidate consistently demonstrate real experience when probed?**

Instead of judging how well something is written, this system evaluates:

* **Depth of knowledge**
* **Consistency across answers**
* **Improvement under follow-ups**

## 🧩 System Architecture

```text
Profile Signals + LLM Answer Analysis + RL Decision Agent
→ Pass / Flag / Probe
```

---

## ⚙️ Components

### 📈 1. Profile Analysis

* Detects abnormal skill growth patterns
* Outputs **Profile Anomaly Score**

---

### 🧠 2. LLM Signal Extraction

Extracts structured signals from answers:

* Tools & concepts
* Reasoning & failures
* Specificity
* Project consistency

---

### 🔍 3. Feature Engineering

* **Depth** → technical richness
* **Consistency** → same experience across answers
* **Progression** → improvement after follow-up

---

### 🤖 4. Reinforcement Learning Agent

Models evaluation like an interviewer:

* **State** → `[profile, depth, consistency, progression, uncertainty]`
* **Actions** → `PASS`, `FLAG`, `PROBE`
* **Goal** → learn when to trust vs probe vs reject

---

### 🔁 5. Interactive Evaluation

```text
Initial Answers → Evaluate → (if uncertain) Probe → Final Decision
```

---

## 🖥️ Demo (Streamlit)

* 📈 Skill growth visualization
* 🧠 LLM-extracted signals
* 🔍 Feature breakdown
* 🔁 Adaptive follow-up question
* ✅ Final decision with reasoning

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Stable-Baselines3 (PPO)
* OpenAI API
* Pandas / NumPy

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
```

```bash
# Train RL agent
python src/train.py
```

```bash
# Run app
streamlit run app.py
```

---

## 📁 Structure

```text
.
├── app.py
├── data/
├── models/
├── src/
│   ├── env.py
│   ├── train.py
│   ├── feature_engineering.py
│   ├── llm_extractor.py
```

---

## 💡 Key Insight

> The system does not detect AI usage.
> It evaluates whether a candidate can **consistently support their claimed expertise under deeper questioning**.

---

## ⚠️ Limitations

* Trained on synthetic data
* Simplified profile signals
* LLM extraction depends on prompt quality

---

## 🔮 Future Work

* Real interview data
* Multi-modal signals (audio/video)
* Fully RL-driven decision pipeline

---

## 🤝 Summary

This project models how a **thoughtful interviewer evaluates expertise** focusing on **behavior under pressure**, not just surface answers.
