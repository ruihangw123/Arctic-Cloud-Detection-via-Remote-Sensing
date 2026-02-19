"""
Reader-friendly labels for variables displayed in figures.
Maps technical variable names to human-readable labels.
"""

VAR_LABELS = {
    # PECARN rule variables
    "GCSTotal": "GCS total",
    "AMS": "Altered mental status",
    "SFxPalp": "Palpable skull fracture",
    "Vomit": "Vomiting",
    "LOCSeparate": "Loss of consciousness",
    "LocLen": "LOC duration",
    "High_impact_InjSev": "Injury mechanism severity",
    "HASeverity": "Headache severity",
    "Hema": "Hematoma",
    "HemaLoc": "Hematoma location",
    "ActNorm": "Acting normally",
    "SFxBas": "Basilar skull fracture",
    "AgeTwoPlus": "Age ≥ 2 years",
    "AgeinYears": "Age (years)",
    # Other common variables
    "PosCT": "Positive CT finding",
    "ciTBI": "Clinically important TBI",
    "CTDone": "CT performed",
    "AgeInMonth": "Age (months)",
    "SFxPalpDepress": "Skull fx: palpable depression",
    "SFxBasHem": "Skull fx: basilar hemorrhage",
    "SFxBasPer": "Skull fx: basilar perforation",
    "SFxBasRhi": "Skull fx: basilar rhinorrhea",
    "SFxBasRet": "Skull fx: basilar retroauricular",
    "SFxBasOto": "Skull fx: basilar otorrhea",
    "SeizLen": "Seizure length",
    "SeizOccur": "Seizure occurrence",
    "CTSedAgitate": "CT sedation: agitation",
    "CTSedAge": "CT sedation: age",
}


def get_label(var: str) -> str:
    """Return reader-friendly label for a variable, or a cleaned version if not in map."""
    if var in VAR_LABELS:
        return VAR_LABELS[var]
    # Fallback: replace underscores with spaces
    return var.replace("_", " ").strip()
