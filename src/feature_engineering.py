import numpy as np

# -----------------------------
# PROFILE FEATURE
# -----------------------------
def compute_profile_anomaly(profile):
    skill_counts = [len(p["skills"]) for p in profile]
    deltas = np.diff(skill_counts)
    return max(deltas) / max(skill_counts) if len(skill_counts) > 1 else 0


# -----------------------------
# DEPTH (PRIMARY SIGNAL)
# -----------------------------
def compute_depth_from_llm(outputs):
    scores = []

    for o in outputs:
        score = o.get("specificity", 0.3)

        if o.get("has_reasoning"):
            score += 0.2

        if o.get("has_failure"):
            score += 0.2

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores)


# -----------------------------
# CONSISTENCY (DEPTH-AWARE)
# -----------------------------
def compute_consistency_from_llm(outputs, depth):
    signals = []

    for o in outputs:
        combined = set()

        # tools
        if o.get("tools"):
            combined.update([t.lower() for t in o["tools"]])

        # concepts
        if o.get("concepts"):
            combined.update([c.lower() for c in o["concepts"]])

        # anchor words
        if o.get("project_anchor"):
            combined.update(o["project_anchor"].lower().split())

        if combined:
            signals.append(combined)

    if len(signals) < 2:
        return 0.5

    overlap_count = 0
    comparisons = 0

    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            if len(signals[i].intersection(signals[j])) > 0:
                overlap_count += 1
            comparisons += 1

    score = overlap_count / comparisons
    consistency = 1.0 if score > 0.3 else 0.3

    # 🔥 depth-aware penalty
    if depth < 0.4:
        consistency *= 0.5

    return consistency


# -----------------------------
# PROGRESSION (DEPTH-AWARE)
# -----------------------------
def compute_progression(scores, depth):
    if len(scores) < 2:
        return 0.5

    # -----------------------------
    # SPECIAL CASE: FOLLOW-UP EXISTS
    # -----------------------------
    if len(scores) >= 4:
        # compare last answer vs best previous answer
        prev_best = max(scores[:-1])
        improvement = scores[-1] - prev_best

        if improvement > 0.2:
            return 1.0  # strong improvement after probe
        elif improvement > 0.05:
            return 0.8  # moderate improvement

    # -----------------------------
    # ORIGINAL LOGIC (unchanged)
    # -----------------------------
    avg = sum(scores) / len(scores)

    if avg > 0.6:
        return 1.0

    variance = max(scores) - min(scores)

    if variance < 0.2 and avg > 0.5:
        return 0.8

    if depth < 0.4:
        return 0.3

    return 0.3

# -----------------------------
# FINAL EXPERTISE SCORE
# -----------------------------
def compute_expertise_score(depth, consistency, progression):
    return (
        0.5 * depth +
        0.3 * consistency +
        0.2 * progression
    )