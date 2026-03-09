"""
Clinical scoring tools for frailty assessment.
These are deterministic calculators used as LangGraph tool nodes.
"""

from backend.models import CFSScore, KatzScore, SPPBScore, FrailtyTier


# --- CFS labels (1-9) ---
CFS_LABELS = {
    1: "Very Fit",
    2: "Well",
    3: "Managing Well",
    4: "Vulnerable",
    5: "Mildly Frail",
    6: "Moderately Frail",
    7: "Severely Frail",
    8: "Very Severely Frail",
    9: "Terminally Ill",
}


def score_cfs(score: int, notes: str = "") -> CFSScore:
    """Return a CFSScore model from a raw 1-9 integer."""
    if score < 1 or score > 9:
        raise ValueError("CFS score must be between 1 and 9.")
    return CFSScore(score=score, label=CFS_LABELS[score], notes=notes or None)


def score_katz(
    bathing: bool,
    dressing: bool,
    toileting: bool,
    transferring: bool,
    continence: bool,
    feeding: bool,
) -> KatzScore:
    """Return a KatzScore from six boolean ADL indicators (True = independent)."""
    return KatzScore(
        bathing=bathing,
        dressing=dressing,
        toileting=toileting,
        transferring=transferring,
        continence=continence,
        feeding=feeding,
    )


def score_sppb(
    balance_score: int,
    gait_speed_score: int,
    chair_stand_score: int,
    notes: str = "",
) -> SPPBScore:
    """Return an SPPBScore from three sub-scores (each 0-4)."""
    return SPPBScore(
        balance_score=balance_score,
        gait_speed_score=gait_speed_score,
        chair_stand_score=chair_stand_score,
        notes=notes or None,
    )


def classify_frailty(
    cfs: CFSScore,
    katz: KatzScore,
    sppb: SPPBScore,
) -> tuple[FrailtyTier, str]:
    """
    Combine CFS, Katz, and SPPB scores into a frailty tier classification.

    Returns:
        (tier, explanation)
    """
    reasons = []

    # CFS-based primary classification
    if cfs.score >= 7:
        tier: FrailtyTier = "severely-frail"
        reasons.append(f"CFS {cfs.score} ({cfs.label}) indicates severe frailty.")
    elif cfs.score >= 5:
        tier = "frail"
        reasons.append(f"CFS {cfs.score} ({cfs.label}) indicates frailty.")
    elif cfs.score == 4:
        tier = "pre-frail"
        reasons.append(f"CFS {cfs.score} ({cfs.label}) indicates vulnerability.")
    else:
        tier = "robust"
        reasons.append(f"CFS {cfs.score} ({cfs.label}) indicates good health.")

    # Katz modifier
    if katz.total <= 3:
        if tier == "robust":
            tier = "pre-frail"
        elif tier == "pre-frail":
            tier = "frail"
        reasons.append(f"Katz ADL score {katz.total}/6 ({katz.label}) suggests functional dependence.")
    elif katz.total <= 5:
        reasons.append(f"Katz ADL score {katz.total}/6 ({katz.label}) — moderate independence.")
    else:
        reasons.append(f"Katz ADL score {katz.total}/6 — full independence.")

    # SPPB modifier
    if sppb.total <= 3:
        if tier in ("robust", "pre-frail"):
            tier = "frail"
        reasons.append(f"SPPB {sppb.total}/12 ({sppb.label}) — severe physical limitation.")
    elif sppb.total <= 6:
        reasons.append(f"SPPB {sppb.total}/12 ({sppb.label}) — moderate physical limitation.")
    else:
        reasons.append(f"SPPB {sppb.total}/12 ({sppb.label}) — adequate physical performance.")

    explanation = " ".join(reasons)
    return tier, explanation
