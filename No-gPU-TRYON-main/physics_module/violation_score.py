def physics_violation_score(strain_score, pose_score, alpha=0.6, beta=0.4):
    score = alpha * strain_score + beta * pose_score

    if score < 0.1:
        label = "Physically Plausible"
    elif score < 0.3:
        label = "Mild Violation"
    else:
        label = "Severe Violation"

    return score, label
