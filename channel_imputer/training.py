"""Session-level sample construction, splitting, encoding and evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .features import extract_event_features, group_sessions, session_channel

logger = logging.getLogger(__name__)


@dataclass
class SessionSample:
    """One session: the unit of training and imputation."""
    user: str
    session_key: str
    event_indices: List[int] = field(repr=False)
    # 1 x feature_dim journey aggregate — shared object across all of a
    # user's sessions (do not mutate).
    agg: sp.csr_matrix = field(repr=False)
    # n_events x feature_dim, one hashed row per session event (in order).
    seq: sp.csr_matrix = field(repr=False)
    label: Optional[str]
    start_ts: float


def build_session_samples(
    user_journeys: Dict[str, List[dict]],
    feature_dim: int,
    max_seq_length: int,
) -> Tuple[FeatureHasher, List[SessionSample], List[SessionSample]]:
    """
    Extract features once per event, hash them immediately to sparse rows,
    and assemble one sample per session.

    Returns (hasher, labelled_samples, unlabelled_samples).  Labelled
    samples train the model; unlabelled ones are the imputation queue.
    Hashing eagerly means raw feature dicts never accumulate in memory and
    nothing has to be re-extracted at imputation time.
    """
    logger.info("Building session-level samples "
                "(feature_dim=%d, max_seq_length=%d)",
                feature_dim, max_seq_length)
    hasher = FeatureHasher(n_features=feature_dim, input_type='dict')

    labelled: List[SessionSample] = []
    unlabelled: List[SessionSample] = []
    total_events = 0

    for user_num, (user, events) in enumerate(user_journeys.items(), start=1):
        feats = [extract_event_features(e) for e in events]
        user_matrix = hasher.transform(feats).tocsr()
        total_events += len(events)

        # Journey-level aggregate: one vector shared by the user's sessions.
        agg = sp.csr_matrix(user_matrix.sum(axis=0))

        for sess in group_sessions(events):
            indices = sess['event_indices']
            label = session_channel(events, indices)
            sample = SessionSample(
                user=user,
                session_key=sess['session_key'],
                event_indices=indices,
                agg=agg,
                seq=user_matrix[indices[:max_seq_length]],
                label=label,
                start_ts=float(min(events[i]['event_timestamp']
                                   for i in indices)),
            )
            (labelled if label is not None else unlabelled).append(sample)

        if user_num % 50_000 == 0:
            logger.info("  Processed %s users...", f"{user_num:,}")

    logger.info(
        "Sessions: %s labelled (training) | %s unlabelled (to impute) "
        "| %s events total",
        f"{len(labelled):,}", f"{len(unlabelled):,}", f"{total_events:,}",
    )
    if not labelled:
        raise RuntimeError(
            "No labelled sessions found — every session is missing a channel."
        )
    return hasher, labelled, unlabelled


def time_then_user_split(
    samples: List[SessionSample], time_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Time-based split first, then a user-holdout validation subset.

    - Train: sessions starting at or before the cutoff timestamp
    - Time-based validation: sessions starting after the cutoff
    - User-holdout validation: the subset of time-based validation whose
      users never appear in training (measures cold-start generalisation)
    """
    timestamps = np.array([s.start_ts for s in samples], dtype=np.float64)
    cutoff = float(np.quantile(timestamps, time_fraction))

    idx_train = np.where(timestamps <= cutoff)[0]
    idx_val_time = np.where(timestamps > cutoff)[0]

    train_users = {samples[i].user for i in idx_train}
    idx_val_user = np.array(
        [i for i in idx_val_time if samples[i].user not in train_users],
        dtype=int,
    )
    logger.info(
        "Split at cutoff %s: %s train | %s val-time | %s val-user sessions",
        f"{int(cutoff):,}", f"{len(idx_train):,}",
        f"{len(idx_val_time):,}", f"{len(idx_val_user):,}",
    )
    if len(idx_val_user) == 0:
        logger.warning("Val-user set is empty; no users are exclusive "
                       "to the validation window.")
    return idx_train, idx_val_time, idx_val_user, cutoff


def encode_labels(
    samples: List[SessionSample], idx_train: np.ndarray,
) -> Tuple[LabelEncoder, np.ndarray, int, Dict[int, float]]:
    """
    Encode string labels to integers and compute balanced class weights.

    The encoder is fitted on all labelled sessions (so validation classes
    are representable), but class weights come from the training subset
    only — computing them over validation data would leak its distribution
    into training.
    """
    le = LabelEncoder()
    y = le.fit_transform([s.label for s in samples])
    num_classes = len(le.classes_)

    y_train = y[idx_train]
    train_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced', classes=train_classes, y=y_train,
    )
    # Classes absent from the training window get weight 1.0 (they cannot
    # be learned this run; they will still be reported in evaluation).
    class_weight_dict = {int(c): 1.0 for c in range(num_classes)}
    class_weight_dict.update(
        {int(c): float(w) for c, w in zip(train_classes, weights)}
    )
    if len(train_classes) < num_classes:
        missing = set(range(num_classes)) - set(int(c) for c in train_classes)
        logger.warning(
            "Classes absent from the training window: %s",
            [le.classes_[c] for c in sorted(missing)],
        )

    logger.info("Classes (%d): %s", num_classes, list(le.classes_))
    logger.info("Class weights (train-only): %s", class_weight_dict)
    return le, y, num_classes, class_weight_dict


def stack_split(samples: List[SessionSample], indices: np.ndarray,
                y: np.ndarray):
    """Assemble the sparse matrices and label vector for one split."""
    if len(indices) == 0:
        return None, [], np.array([], dtype=np.int64)
    agg = sp.vstack([samples[i].agg for i in indices]).tocsr()
    seq = [samples[i].seq for i in indices]
    return agg, seq, y[indices]


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray,
                         class_names: List[str]) -> dict:
    """Per-class precision/recall/F1 and confusion matrix for a split."""
    labels = np.arange(len(class_names))
    y_pred = probs.argmax(axis=1)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float((y_pred == y_true).mean()) if len(y_true) else None
    return {
        'accuracy': accuracy,
        'n_samples': int(len(y_true)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': list(class_names),
    }
