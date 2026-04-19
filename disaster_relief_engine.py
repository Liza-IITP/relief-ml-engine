"""
Disaster Relief — Adaptive ML Matching Engine

Pipeline Overview:
- Part 1: Historical data generator (2,000 rows, Faker)
- Part 2: Adaptive ML pipeline (XGBoost, SGD residual layer, LightGBM LambdaRank, Hungarian algorithm)
- Part 3: Feature importance reporting
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import uuid
import random
import logging
import warnings
import time
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Any

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
# pip install faker numpy scipy scikit-learn xgboost lightgbm
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from faker import Faker
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  |  %(levelname)-8s  |  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relief_ml")

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)
Faker.seed(42)


# --- Constants ---

CATEGORIES = [
    "food", "water", "shelter", "medical",
    "education", "sanitation", "psychosocial", "livelihood",
]
ZONES  = ["Ward_1", "Ward_2", "Ward_3", "Ward_4", "Ward_5"]
SKILLS = [
    "first_aid", "heavy_lifting", "counseling",
    "driving", "translation", "medical_license",
]

# All feature names in the exact order the models see them.
# Explicit ordering eliminates any subtle column-shuffle bugs.
FEATURE_NAMES = [
    # -- Volunteer features --------------------------------------------------
    "reliability_score",          # F0  -- core signal, Rigged Rule 3
    "historical_completion_rate", # F1
    "avg_response_time_hrs",      # F2
    "total_assignments",          # F3
    "vol_n_skills",               # F4
    # -- Pair / interaction features -----------------------------------------
    "zone_match",                 # F5  -- Rigged Rule 2 (with severity)
    "skill_overlap_ratio",        # F6  -- Rigged Rule 1 (medical_license)
    "has_medical_license",        # F7  -- explicit Rule-1 signal
    "required_medical_license",   # F8  -- explicit Rule-1 signal (need side)
    "skill_surplus",              # F9
    "scarcity_weighted_overlap",  # F10
    # -- Need features -------------------------------------------------------
    "severity",                   # F11 -- Rigged Rule 2 (with zone mismatch)
    "vulnerability_index",        # F12
    "people_affected",            # F13
    "deadline_urgency",           # F14 (1/hours or 0)
    "category_risk_prior",        # F15
    "vol_count_needed",           # F16
    # -- Category one-hot (8) ------------------------------------------------
    "cat_food", "cat_water", "cat_shelter", "cat_medical",
    "cat_education", "cat_sanitation", "cat_psychosocial", "cat_livelihood",
]

CATEGORY_RISK_PRIOR = {
    "medical": 0.90, "water": 0.85, "food": 0.80, "shelter": 0.75,
    "sanitation": 0.65, "psychosocial": 0.55, "livelihood": 0.45,
    "education": 0.35,
}
SKILL_SCARCITY = {
    "medical_license": 1.8, "first_aid": 1.3, "counseling": 1.2,
    "driving": 1.0, "translation": 1.1, "heavy_lifting": 0.9,
}
N_FEATURES = len(FEATURE_NAMES)   # 25


# --- Part 1: Historical Data Generator ---

def _random_skills(k_min: int = 1, k_max: int = 3) -> list:
    return random.sample(SKILLS, random.randint(k_min, k_max))


def _random_need() -> dict:
    cat = random.choice(CATEGORIES)
    return {
        "id":                    str(uuid.uuid4()),
        "category":              cat,
        "zone":                  random.choice(ZONES),
        "severity":              random.randint(1, 5),
        "vulnerability_index":   random.randint(1, 5),
        "people_affected":       random.randint(1, 15),
        "hours_until_deadline":  random.choice([None, 6, 12, 24, 48, 72]),
        "volunteer_count_needed": random.randint(1, 3),
        # 25% of medical needs require medical_license;  10% of others do
        "required_skills": (
            ["medical_license"] + random.sample(
                [s for s in SKILLS if s != "medical_license"],
                random.randint(0, 1),
            )
            if (cat == "medical" and random.random() < 0.25)
            or (cat != "medical" and random.random() < 0.10)
            else random.sample(SKILLS, random.randint(0, 1))
        ),
    }


def _random_volunteer() -> dict:
    skills = _random_skills()
    return {
        "id":               str(uuid.uuid4()),
        "zone":             random.choice(ZONES),
        "skills":           skills,
        "reliability_score": random.randint(30, 100),
        "historical_completion_rate": round(random.uniform(0.40, 1.0), 3),
        "avg_response_time_hrs":      round(random.uniform(0.5, 10.0), 2),
        "total_assignments":          random.randint(0, 60),
        "active": True,
    }


def apply_rigged_rules(need: dict, vol: dict) -> bool:
    """
    The three hidden logical rules that generate the target label.
    The ML must DISCOVER these patterns from correlation alone --
    the rules themselves are never passed as raw booleans to the model.

    Rule 1 -- Skill Gate (medical_license):
        need requires medical_license AND volunteer lacks it
        --> 95% chance need_met = False

    Rule 2 -- Zone x Severity:
        zone mismatch AND severity >= 4
        --> 80% chance need_met = False

    Rule 3 -- Reliability Floor:
        reliability_score < 60
        --> 70% chance need_met = False

    Otherwise: sigmoid on reliability_score drives base probability.
    """
    req_skills = set(need.get("required_skills", []))
    vol_skills = set(vol.get("skills", []))
    zone_match = vol.get("zone") == need.get("zone")
    rel        = vol.get("reliability_score", 50)

    # -- Rule 1: Skill Gate --------------------------------------------------
    if "medical_license" in req_skills and "medical_license" not in vol_skills:
        return random.random() > 0.95    # 5% slip-through (noisy labels)

    # -- Rule 2: Zone x Severity ----------------------------------------------
    if not zone_match and need.get("severity", 1) >= 4:
        return random.random() > 0.80    # 20% slip-through

    # -- Rule 3: Reliability Floor --------------------------------------------
    if rel < 60:
        return random.random() > 0.70    # 30% slip-through

    # -- Base probability: sigmoid on reliability_score -----------------------
    # rel=50 -> p~0.50;  rel=80 -> p~0.82;  rel=100 -> p~0.95
    p = 1.0 / (1.0 + np.exp(-0.08 * (rel - 50)))
    if zone_match:               p = min(p + 0.10, 0.97)
    if req_skills <= vol_skills: p = min(p + 0.07, 0.97)
    return random.random() < p


def generate_historical_data(n: int = 2000) -> list:
    """
    Generate n (need, volunteer, outcome) rows with rigged labels.
    Returns list of record dicts ready for feature extraction.
    """
    records = []
    for _ in range(n):
        need     = _random_need()
        vol      = _random_volunteer()
        need_met = apply_rigged_rules(need, vol)
        records.append({"need": need, "volunteer": vol, "need_met": need_met})

    true_count  = sum(1 for r in records if r["need_met"])
    false_count = n - true_count
    log.info(
        "Historical data | n=%d | need_met=True:%d (%.1f%%)  False:%d (%.1f%%)",
        n, true_count, 100*true_count/n, false_count, 100*false_count/n,
    )
    return records


# --- Feature Engineering ---

def build_feature_vector(need: dict, vol: dict) -> np.ndarray:
    """
    Returns float32 array of shape (N_FEATURES,) = (25,).
    Column order is EXACTLY aligned with FEATURE_NAMES for interpretability.
    """
    req_skills  = set(need.get("required_skills", []))
    vol_skills  = set(vol.get("skills", []))
    overlap     = vol_skills & req_skills
    zone_match  = float(vol.get("zone", "") == need.get("zone", ""))
    cat         = need.get("category", "food")
    hrs         = need.get("hours_until_deadline")
    rel         = vol.get("reliability_score", 50)
    scarcity_overlap = sum(SKILL_SCARCITY.get(s, 1.0) for s in overlap)

    # Category one-hot
    cat_oh = np.zeros(8, dtype=np.float32)
    if cat in CATEGORIES:
        cat_oh[CATEGORIES.index(cat)] = 1.0

    vec = np.array([
        # Volunteer
        rel / 100.0,
        vol.get("historical_completion_rate", 0.7),
        1.0 / (1.0 + vol.get("avg_response_time_hrs", 4.0)),   # inverted: lower RT better
        np.log1p(vol.get("total_assignments", 0)) / np.log1p(60),
        len(vol_skills) / 6.0,
        # Pair / interaction
        zone_match,
        len(overlap) / max(len(req_skills), 1),                 # overlap ratio
        float("medical_license" in vol_skills),                 # Rule-1 signal
        float("medical_license" in req_skills),                 # Rule-1 need side
        (len(vol_skills) - len(req_skills)) / 6.0,              # surplus
        scarcity_overlap / max(len(req_skills) * 1.8, 1.0),
        # Need
        need.get("severity", 3) / 5.0,
        need.get("vulnerability_index", 3) / 5.0,
        min(need.get("people_affected", 1), 15) / 15.0,
        1.0 / hrs if (hrs and hrs > 0) else 0.0,
        CATEGORY_RISK_PRIOR.get(cat, 0.5),
        need.get("volunteer_count_needed", 1) / 3.0,
    ], dtype=np.float32)

    return np.concatenate([vec, cat_oh])   # (25,)


def records_to_Xy(records: list) -> tuple:
    X = np.stack([build_feature_vector(r["need"], r["volunteer"]) for r in records])
    y = np.array([int(r["need_met"]) for r in records], dtype=np.int32)
    return X.astype(np.float32), y


# --- Part 2A: XGBoost Base Classifier ---

class XGBoostOutcomePredictor:
    """
    Predicts P(need_met=True) for a (need, volunteer) pair.
    Trained once on full historical dataset; serves as the stable base layer
    whose predictions the SGD residual corrector learns to refine online.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.80,
            colsample_bytree=0.75,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=1.0,   # updated after seeing class ratio
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=30,
        )
        self.feature_names = FEATURE_NAMES
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> None:
        # Compute positive weight from training split
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        self.model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.is_fitted = True

        # Validation metrics
        y_prob = self.model.predict_proba(X_val)[:, 1]
        auc    = roc_auc_score(y_val, y_prob)
        ap     = average_precision_score(y_val, y_prob)
        log.info("XGBoost | Val AUC=%.4f  AP=%.4f  best_iter=%d",
                 auc, ap, self.model.best_iteration)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance_report(self) -> list:
        """Returns sorted (feature_name, importance) by 'gain'."""
        scores = self.model.get_booster().get_score(importance_type="gain")
        named = []
        for raw_key, val in scores.items():
            idx = int(raw_key.replace("f", ""))
            named.append((FEATURE_NAMES[idx], val))
        named.sort(key=lambda x: x[1], reverse=True)
        return named


# --- Part 2B: SGD Online Residual Corrector ---

class SGDResidualCorrector:
    """
    Lightweight online layer trained on residuals:
        residual = true_label - xgb_probability

    Uses partial_fit to simulate real-time learning from incoming outcome
    batches without full model retraining.

    Inspired by: Uber's lambda-annealed coefficient update loop where a fast,
    stateful linear model absorbs short-term distribution drift while the
    heavy ensemble captures long-term structure.

    alpha=0.85 in ensemble_score heavily favours XGBoost (stable, data-rich);
    SGD provides a small but meaningful online correction. As SGD accumulates
    more updates, alpha could be annealed toward 0.70.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.sgd    = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=1,
            warm_start=True,
            random_state=42,
            class_weight=None,
        )
        self.is_fitted  = False
        self.n_updates  = 0
        self.corrections = []    # track drift magnitude over time

    def partial_fit(
        self,
        X_batch:   np.ndarray,
        y_batch:   np.ndarray,
        xgb_probs: np.ndarray,
    ) -> None:
        """
        Fit on residual signal: samples where XGBoost was confidently wrong
        carry higher learning signal (|residual| > 0.3 threshold).
        """
        residuals  = y_batch.astype(np.float32) - xgb_probs
        hard_cases = np.abs(residuals) > 0.3
        if hard_cases.sum() < 5:
            return   # batch too clean to learn from

        X_hard = X_batch[hard_cases].astype(np.float64)
        y_hard = y_batch[hard_cases]

        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_hard)
            self.sgd.partial_fit(X_scaled, y_hard, classes=np.array([0, 1]))
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_hard)
            self.sgd.partial_fit(X_scaled, y_hard)

        self.n_updates += 1
        drift = float(np.mean(np.abs(residuals[hard_cases])))
        self.corrections.append(drift)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X), dtype=np.float32)
        X_scaled = self.scaler.transform(X.astype(np.float64))
        return self.sgd.predict_proba(X_scaled)[:, 1].astype(np.float32)

    def ensemble_score(
        self,
        X:        np.ndarray,
        xgb_prob: np.ndarray,
        alpha:    float = 0.85,
    ) -> np.ndarray:
        """Blend XGBoost and SGD predictions."""
        if not self.is_fitted:
            return xgb_prob
        sgd_prob = self.predict_proba(X)
        return alpha * xgb_prob + (1 - alpha) * sgd_prob


# --- Part 2C: LightGBM LambdaRank Volunteer Ranker ---

class LambdaRankVolunteerRanker:
    """
    LightGBM configured for LambdaRank (NDCG@1,3,5 optimisation).

    Why LambdaRank over pointwise regression?
    -----------------------------------------
    Matching is fundamentally an ordering problem. LambdaRank directly
    optimises NDCG -- the metric that rewards putting the best volunteer
    at rank 1 more than getting an average score right. This matches the
    real objective: the top pick matters most.

    Relevance labels (0-3):
        3 = perfect: zone match + all skills + reliability >= 85
        2 = good:    zone match + all skills + reliability >= 65
        1 = marginal: zone match OR all skills satisfied
        0 = poor:    neither condition met
    """

    def __init__(self):
        self.model     = None
        self.n_updates = 0

    def _assign_relevance(self, need: dict, vol: dict) -> int:
        zone_ok  = vol.get("zone") == need.get("zone")
        req      = set(need.get("required_skills", []))
        skill_ok = req <= set(vol.get("skills", []))
        rel      = vol.get("reliability_score", 50)
        if zone_ok and skill_ok and rel >= 85: return 3
        if zone_ok and skill_ok and rel >= 65: return 2
        if zone_ok or skill_ok:               return 1
        return 0

    def build_dataset(
        self,
        needs: list,
        volunteer_pool: list,
        max_candidates_per_need: int = 15,
    ) -> tuple:
        """
        Build LTR arrays: X (pairs), y (relevance), groups (per-need sizes).
        Each need forms one query group; volunteers are the documents.
        """
        rows, labels, groups = [], [], []
        for need in needs:
            zone_vols = [
                v for v in volunteer_pool
                if v.get("zone") == need.get("zone") and v.get("active", True)
            ]
            # Pad with cross-zone volunteers so every need has candidates
            if len(zone_vols) < 5:
                zone_vols += [v for v in volunteer_pool if v not in zone_vols]
            candidates = zone_vols[:max_candidates_per_need]
            if not candidates:
                continue
            for vol in candidates:
                rows.append(build_feature_vector(need, vol))
                labels.append(self._assign_relevance(need, vol))
            groups.append(len(candidates))

        if not rows:
            return np.empty((0, N_FEATURES)), np.empty(0), np.empty(0)
        return (
            np.stack(rows).astype(np.float32),
            np.array(labels, dtype=np.int32),
            np.array(groups, dtype=np.int32),
        )

    def fit(self, needs: list, volunteer_pool: list) -> None:
        X, y, groups = self.build_dataset(needs, volunteer_pool)
        if len(X) == 0:
            log.warning("LambdaRank | no training data")
            return

        train_ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
        params = {
            "objective":        "lambdarank",
            "metric":           "ndcg",
            "ndcg_eval_at":     [1, 3, 5],
            "learning_rate":    0.05,
            "num_leaves":       31,
            "min_data_in_leaf": 5,
            "lambda_l2":        0.1,
            "verbosity":        -1,
            "seed":             42,
            "label_gain":       [0, 1, 3, 7],   # exponential gain for levels 0-3
        }
        self.model = lgb.train(
            params, train_ds,
            num_boost_round=200,
            valid_sets=[train_ds],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        log.info(
            "LambdaRank | fit complete | best_iter=%d",
            self.model.best_iteration,
        )

    def rank(self, need: dict, candidates: list) -> list:
        """Return [(score, volunteer), ...] sorted descending."""
        if not candidates:
            return []
        if self.model is None:
            return sorted(
                [(v.get("reliability_score", 50) / 100.0, v) for v in candidates],
                reverse=True,
            )
        X = np.stack([build_feature_vector(need, v) for v in candidates]).astype(np.float32)
        scores = self.model.predict(X)
        return sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)

    def incremental_update(
        self,
        needs: list,
        volunteer_pool: list,
        extra_rounds: int = 25,
    ) -> None:
        """Continue boosting from current model state (online update)."""
        if self.model is None:
            self.fit(needs, volunteer_pool)
            return
        X, y, groups = self.build_dataset(needs, volunteer_pool)
        if len(X) == 0:
            return
        train_ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "ndcg_eval_at": [1, 3], "learning_rate": 0.02,
            "num_leaves": 31, "verbosity": -1, "seed": 42,
            "label_gain": [0, 1, 3, 7],
        }
        self.model = lgb.train(
            params, train_ds,
            num_boost_round=extra_rounds,
            init_model=self.model,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        self.n_updates += 1
        log.info("LambdaRank | incremental update #%d (+%d rounds)",
                 self.n_updates, extra_rounds)

    def feature_importance_report(self) -> list:
        if self.model is None:
            return []
        imps = self.model.feature_importance(importance_type="gain")
        named = list(zip(FEATURE_NAMES, imps.tolist()))
        named.sort(key=lambda x: x[1], reverse=True)
        return named


# --- Part 2D: Hungarian Global Optimizer ---

class HungarianOptimalAssigner:
    """
    Globally optimal volunteer-to-need assignment using
    scipy.optimize.linear_sum_assignment (Hungarian algorithm).

    Given a batch of N needs and M volunteers:
      1. Score every valid (need, volunteer) pair with the ensemble model.
      2. Build cost matrix (negate scores for minimisation).
      3. Hungarian finds the assignment maximising total ensemble score.
      4. Hard constraints (zone, skills, active) set invalid cells to penalty 1e6.

    Complexity: O(N^3) -- acceptable for N < 1,000 per batch.
    For larger batches, use auction algorithm or block decomposition.
    """

    def assign(
        self,
        needs:     list,
        volunteers: list,
        xgb_model: XGBoostOutcomePredictor,
        sgd_model: SGDResidualCorrector,
        ranker:    LambdaRankVolunteerRanker,
    ) -> list:
        active_vols = [v for v in volunteers if v.get("active", True)]
        n_vols      = len(active_vols)

        # Expand needs to slots (a need wanting 3 volunteers -> 3 rows)
        need_slots = []
        for need in needs:
            for _ in range(need.get("volunteer_count_needed", 1)):
                need_slots.append(need)

        n_slots = len(need_slots)
        cost = np.full((n_slots, n_vols), fill_value=1e6, dtype=np.float64)

        # Collect valid pairs and batch-score them
        valid_pairs = []
        for si, need in enumerate(need_slots):
            for vi, vol in enumerate(active_vols):
                zone_ok  = vol.get("zone") == need.get("zone")
                req      = set(need.get("required_skills", []))
                skill_ok = req <= set(vol.get("skills", []))
                if zone_ok and skill_ok:
                    valid_pairs.append((si, vi))

        if valid_pairs:
            X_pairs = np.stack([
                build_feature_vector(need_slots[si], active_vols[vi])
                for si, vi in valid_pairs
            ]).astype(np.float32)

            xgb_probs = xgb_model.predict_proba(X_pairs)
            scores    = sgd_model.ensemble_score(X_pairs, xgb_probs)

            for (si, vi), score in zip(valid_pairs, scores.tolist()):
                cost[si, vi] = 1.0 - score   # negate for minimisation

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments      = []
        used_vol_indices = set()

        for slot_idx, vol_idx in zip(row_ind, col_ind):
            if cost[slot_idx, vol_idx] >= 1e5:
                continue   # no valid match for this slot
            if vol_idx in used_vol_indices:
                continue   # volunteer already assigned this batch

            need = need_slots[slot_idx]
            vol  = active_vols[vol_idx]

            ranked     = ranker.rank(need, [vol])
            rank_score = ranked[0][0] if ranked else 0.0

            assignments.append({
                "id":              str(uuid.uuid4()),
                "need_id":         need["id"],
                "volunteer_id":    vol["id"],
                "status":          "pending",
                "assigned_by":     "system",
                "_xgb_score":      round(float(1.0 - cost[slot_idx, vol_idx]), 4),
                "_lambdarank_score": round(rank_score, 4),
                "_zone_match":     vol.get("zone") == need.get("zone"),
                "_skills_met":     set(need.get("required_skills", [])) <= set(vol.get("skills", [])),
            })
            used_vol_indices.add(vol_idx)

        return assignments


# --- Part 3: "Judge Flex" Feature Importance Report ---

# Mapping rigged rules to the proxy features the model should upweight
RULE_FEATURE_MAP = {
    "Rule 1 -- Skill Gate (medical_license)": [
        "has_medical_license", "required_medical_license", "skill_overlap_ratio",
    ],
    "Rule 2 -- Zone x Severity": [
        "zone_match", "severity",
    ],
    "Rule 3 -- Reliability Floor": [
        "reliability_score", "historical_completion_rate",
    ],
}


def print_feature_importance_report(
    xgb_imps: list,
    lgb_imps: list,
) -> None:
    """
    Formatted console report proving the model discovered the hidden rules.
    The "Gain" metric measures total reduction in impurity (XGBoost) or
    total NDCG improvement (LightGBM) contributed by each feature --
    the higher the gain, the more the model relied on that feature.
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print("  *  FEATURE IMPORTANCE -- JUDGE FLEX REPORT  *")
    print("  Proving the ML engine discovered the 3 hidden rigged rules.")
    print(SEP)

    # -- XGBoost importance (gain) --------------------------------------------
    print("\n  XGBoost -- Feature Importance (Gain)")
    print("  Gain: total reduction in impurity gained by this feature\n")

    max_gain = xgb_imps[0][1] if xgb_imps else 1.0
    for rank, (name, score) in enumerate(xgb_imps[:15], 1):
        bar_len = int(40 * score / max_gain)
        bar     = "#" * bar_len + "." * (40 - bar_len)
        tag     = ""
        for rule, features in RULE_FEATURE_MAP.items():
            if name in features:
                rule_num = rule.split("--")[0].strip()
                tag = f"  <-- {rule_num}"
                break
        print(f"  {rank:>2}. {name:<35} {score:>9.1f}  {bar}{tag}")

    # -- LightGBM LambdaRank importance ---------------------------------------
    print(f"\n{SEP2}")
    print("\n  LightGBM LambdaRank -- Feature Importance (Gain)")
    print("  Gain: total NDCG improvement contributed by this feature\n")

    max_lgb = lgb_imps[0][1] if lgb_imps else 1.0
    for rank, (name, score) in enumerate(lgb_imps[:15], 1):
        bar_len = int(40 * score / max_lgb)
        bar     = "#" * bar_len + "." * (40 - bar_len)
        tag     = ""
        for rule, features in RULE_FEATURE_MAP.items():
            if name in features:
                rule_num = rule.split("--")[0].strip()
                tag = f"  <-- {rule_num}"
                break
        print(f"  {rank:>2}. {name:<35} {score:>9.1f}  {bar}{tag}")

    # -- Rule Discovery Summary -----------------------------------------------
    print(f"\n{SEP}")
    print("  RULE DISCOVERY SUMMARY")
    print(SEP)

    xgb_dict = dict(xgb_imps)
    lgb_dict = dict(lgb_imps)

    for rule, features in RULE_FEATURE_MAP.items():
        xgb_scores = [xgb_dict.get(f, 0.0) for f in features]
        lgb_scores = [lgb_dict.get(f, 0.0) for f in features]
        xgb_top7   = {name for name, _ in xgb_imps[:7]}
        lgb_top7   = {name for name, _ in lgb_imps[:7]}
        discovered  = any(f in xgb_top7 or f in lgb_top7 for f in features)
        indicator   = "[DISCOVERED]" if discovered else "[WEAK SIGNAL]"
        print(f"\n  {indicator}  {rule}")
        print(f"     Proxy features : {', '.join(features)}")
        if xgb_scores:
            xgb_max_feat = features[int(np.argmax(xgb_scores))]
            print(f"     XGBoost top proxy    : {xgb_max_feat:<32} gain={max(xgb_scores):>8.1f}")
        if lgb_scores:
            lgb_max_feat = features[int(np.argmax(lgb_scores))]
            print(f"     LambdaRank top proxy : {lgb_max_feat:<32} gain={max(lgb_scores):>8.1f}")

    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# WEBHOOK & INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_models() -> dict:
    """
    Generate mock historical data and train models once, returning them to be held in memory.
    """
    log.info("Initializing models on server startup...")
    records = generate_historical_data(2000)
    X_all, y_all = records_to_Xy(records)
    
    split = int(0.80 * len(records))
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    hist_volunteers = [r["volunteer"] for r in records]
    hist_needs      = [r["need"]      for r in records]

    xgb_model = XGBoostOutcomePredictor()
    xgb_model.fit(X_tr, y_tr, X_val, y_val)

    sgd_model = SGDResidualCorrector()
    batch_size = 100
    for i in range(3):
        lo = i * batch_size
        hi = lo + batch_size
        Xb = X_val[lo:hi]
        yb = y_val[lo:hi]
        if len(Xb) == 0: break
        xgb_probs = xgb_model.predict_proba(Xb)
        sgd_model.partial_fit(Xb, yb, xgb_probs)

    ranker = LambdaRankVolunteerRanker()
    sample_needs = random.sample(hist_needs, min(300, len(hist_needs)))
    ranker.fit(sample_needs, hist_volunteers)

    log.info("Models initialized and held in memory.")
    return {
        "xgb_model": xgb_model,
        "sgd_model": sgd_model,
        "ranker": ranker
    }

def process_live_webhook(new_need: dict, active_volunteers: list, models: dict) -> list:
    """
    Takes a new need and active volunteers, scores them via models, and returns optimal assignments.
    """
    log.info(f"Processing webhook for need: {new_need.get('id')} with {len(active_volunteers)} active volunteers")
    
    # If no volunteers, return empty list
    if not active_volunteers:
        return []

    assigner = HungarianOptimalAssigner()
    assignments = assigner.assign(
        [new_need], 
        active_volunteers, 
        models["xgb_model"], 
        models["sgd_model"], 
        models["ranker"]
    )
    return assignments


# --- Orchestrator ---

def run_pipeline() -> None:
    t0 = time.perf_counter()
    log.info("=" * 72)
    log.info("  DISASTER RELIEF PHASE 2 ML PIPELINE -- START")
    log.info("=" * 72)

    # -- Part 1: Generate rigged historical data ------------------------------
    log.info("Part 1 | Generating 2,000 rigged historical records ...")
    records = generate_historical_data(2000)

    X_all, y_all = records_to_Xy(records)
    split    = int(0.80 * len(records))
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    hist_volunteers = [r["volunteer"] for r in records]
    hist_needs      = [r["need"]      for r in records]

    # -- Part 2A: XGBoost -----------------------------------------------------
    log.info("Part 2A | Training XGBoost outcome predictor ...")
    xgb_model = XGBoostOutcomePredictor()
    xgb_model.fit(X_tr, y_tr, X_val, y_val)

    # -- Part 2B: SGD residual corrector (warm-start on validation batches) ---
    log.info("Part 2B | Warming SGD residual corrector on 3 outcome batches ...")
    sgd_model  = SGDResidualCorrector()
    batch_size = 100
    for i in range(3):
        lo = i * batch_size
        hi = lo + batch_size
        Xb = X_val[lo:hi]
        yb = y_val[lo:hi]
        if len(Xb) == 0:
            break
        xgb_probs = xgb_model.predict_proba(Xb)
        sgd_model.partial_fit(Xb, yb, xgb_probs)
        drift_str = (
            f"{sgd_model.corrections[-1]:.3f}"
            if sgd_model.corrections else "n/a"
        )
        log.info("  SGD batch %d | n_updates=%d | hard_case_drift=%s",
                 i+1, sgd_model.n_updates, drift_str)

    # -- Part 2C: LightGBM LambdaRank -----------------------------------------
    log.info("Part 2C | Training LambdaRank volunteer ranker ...")
    ranker       = LambdaRankVolunteerRanker()
    sample_needs = random.sample(hist_needs, min(300, len(hist_needs)))
    ranker.fit(sample_needs, hist_volunteers)

    # -- Part 2D: Generate 20 new open needs + Hungarian assignment -----------
    log.info("Part 2D | Generating 20 new open needs for live matching ...")
    new_needs = [_random_need() for _ in range(20)]
    # Inject one guaranteed medical_license need for edge-case validation
    new_needs.append({
        "id": str(uuid.uuid4()), "category": "medical",
        "zone": "Ward_2", "severity": 5, "vulnerability_index": 5,
        "people_affected": 10, "hours_until_deadline": 6,
        "volunteer_count_needed": 1, "required_skills": ["medical_license"],
    })

    # Fresh volunteer pool
    live_volunteers = [_random_volunteer() for _ in range(60)]
    # Guarantee one qualified doctor in Ward_2
    live_volunteers.append({
        "id": str(uuid.uuid4()), "zone": "Ward_2",
        "skills": ["medical_license", "first_aid"],
        "reliability_score": 92,
        "historical_completion_rate": 0.95,
        "avg_response_time_hrs": 0.5,
        "total_assignments": 30,
        "active": True,
    })

    log.info("Part 2D | Running Hungarian optimal assignment ...")
    assigner    = HungarianOptimalAssigner()
    assignments = assigner.assign(
        new_needs, live_volunteers, xgb_model, sgd_model, ranker,
    )
    log.info("Part 2D | %d assignments created for %d needs",
             len(assignments), len(new_needs))

    # -- Online feedback loop: one more SGD + LambdaRank update after live run
    log.info("Feedback | Simulating online outcome feedback after live run ...")
    need_map = {n["id"]: n for n in new_needs}
    vol_map  = {v["id"]: v for v in live_volunteers}
    live_pairs = [
        (need_map[a["need_id"]], vol_map[a["volunteer_id"]])
        for a in assignments
        if a["need_id"] in need_map and a["volunteer_id"] in vol_map
    ]
    if live_pairs:
        X_live    = np.stack([build_feature_vector(n, v) for n, v in live_pairs]).astype(np.float32)
        xgb_live  = xgb_model.predict_proba(X_live)
        y_live    = (xgb_live > 0.50).astype(np.int32)
        sgd_model.partial_fit(X_live, y_live, xgb_live)
        ranker.incremental_update(new_needs, live_volunteers, extra_rounds=25)

    # -- Part 3: Feature Importance Report ------------------------------------
    xgb_imps = xgb_model.feature_importance_report()
    lgb_imps = ranker.feature_importance_report()
    print_feature_importance_report(xgb_imps, lgb_imps)

    # -- Validation classification report ------------------------------------
    log.info("Validation | Held-out classification report:")
    y_pred = (xgb_model.predict_proba(X_val) > 0.50).astype(int)
    print(classification_report(y_val, y_pred, target_names=["False (unmet)", "True (met)"]))

    # -- Export model_state.json ---------------------------------------------
    elapsed = round(time.perf_counter() - t0, 3)
    val_auc = round(roc_auc_score(y_val, xgb_model.predict_proba(X_val)), 4)

    model_state = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "pipeline_time_s": elapsed,
        "historical_data": {
            "n_records":    2000,
            "train_split":  1600,
            "val_split":    400,
            "class_balance": {
                "need_met_true":  int(y_all.sum()),
                "need_met_false": int((y_all == 0).sum()),
            },
            "rigged_rules": [
                "Rule 1: medical_license gate -- 95% False when license missing",
                "Rule 2: zone mismatch x severity>=4 -- 80% False",
                "Rule 3: reliability_score < 60 -- 70% False",
            ],
        },
        "xgboost": {
            "type":         "XGBClassifier (binary:logistic)",
            "n_estimators": 400,
            "val_auc":      val_auc,
            "top_10_features_by_gain": [
                {"feature": n, "gain": round(g, 2)} for n, g in xgb_imps[:10]
            ],
        },
        "sgd_residual": {
            "type":              "SGDClassifier (log_loss + elasticnet)",
            "n_updates":         sgd_model.n_updates,
            "is_fitted":         sgd_model.is_fitted,
            "correction_drift":  [round(c, 4) for c in sgd_model.corrections],
            "ensemble_alpha":    0.85,
        },
        "lambdarank": {
            "type":              "LightGBM LambdaRank (NDCG@1,3,5)",
            "best_iteration":    getattr(ranker.model, "best_iteration", None),
            "n_incremental_updates": ranker.n_updates,
            "top_10_features_by_gain": [
                {"feature": n, "gain": round(g, 2)} for n, g in lgb_imps[:10]
            ],
        },
        "hungarian_assignment": {
            "type":          "scipy.optimize.linear_sum_assignment",
            "n_needs":       len(new_needs),
            "n_volunteers":  len(live_volunteers),
            "n_assignments": len(assignments),
            "sample":        assignments[:5],
        },
        "feature_schema": FEATURE_NAMES,
        "rule_discovery_map": RULE_FEATURE_MAP,
    }

    with open("model_state.json", "w", encoding="utf-8") as f:
        json.dump(model_state, f, indent=2, default=str)

    log.info("=" * 72)
    log.info("  PIPELINE COMPLETE | %.3f s | model_state.json written", elapsed)
    log.info("  Val AUC=%.4f | Assignments=%d/%d needs", val_auc, len(assignments), len(new_needs))
    log.info("=" * 72)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
