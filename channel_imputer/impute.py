"""Impute channels for unattributed sessions using the trained model."""

import logging
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

from .config import PipelineConfig
from .model import densify_batch
from .training import SessionSample

logger = logging.getLogger(__name__)


def impute_missing_channels(
    unlabelled: List[SessionSample],
    user_journeys: Dict[str, List[dict]],
    model,
    label_encoder,
    cfg: PipelineConfig,
) -> dict:
    """
    Predict one channel per unattributed session and apply it to every
    event in that session, so a session never ends up with conflicting
    imputed channels.

    Sessions whose top predicted probability falls below
    cfg.confidence_threshold keep their original (missing) channel — a
    wrong confident-sounding label is worse for attribution than an honest
    gap.  Returns a summary dict for the run report.
    """
    total = len(unlabelled)
    logger.info("Imputing channels for %s unattributed sessions "
                "(confidence threshold: %.2f)",
                f"{total:,}", cfg.confidence_threshold)
    if total == 0:
        return {'sessions_to_impute': 0, 'sessions_imputed': 0,
                'sessions_below_threshold': 0, 'events_imputed': 0,
                'mean_confidence': None}

    applied = 0
    below_threshold = 0
    events_imputed = 0
    confidences = np.empty(total, dtype=np.float64)

    batch_size = cfg.inference_batch_size
    for start in range(0, total, batch_size):
        batch = unlabelled[start:start + batch_size]
        inputs = densify_batch(
            sp.vstack([s.agg for s in batch]).tocsr(),
            [s.seq for s in batch],
            cfg.max_seq_length, cfg.feature_dim,
        )
        probs = model.predict(inputs, batch_size=batch_size, verbose=0)
        pred_idx = probs.argmax(axis=1)
        pred_channels = label_encoder.inverse_transform(pred_idx)
        pred_conf = probs.max(axis=1)

        for offset, (sample, channel, conf) in enumerate(
                zip(batch, pred_channels, pred_conf)):
            conf = float(conf)
            confidences[start + offset] = conf
            if conf < cfg.confidence_threshold:
                below_threshold += 1
                continue
            for i in sample.event_indices:
                event = user_journeys[sample.user][i]
                event['final_channel'] = channel
                event['was_imputed'] = True
                event['imputation_confidence'] = conf
                events_imputed += 1
            applied += 1

        done = min(start + len(batch), total)
        if (start // batch_size) % 20 == 0:
            logger.info("  Imputed %s / %s sessions...",
                        f"{done:,}", f"{total:,}")

    summary = {
        'sessions_to_impute': total,
        'sessions_imputed': applied,
        'sessions_below_threshold': below_threshold,
        'events_imputed': events_imputed,
        'mean_confidence': float(confidences.mean()),
        'median_confidence': float(np.median(confidences)),
        'min_confidence': float(confidences.min()),
    }
    logger.info(
        "Imputation complete: %s sessions applied (%s events), "
        "%s left unimputed below threshold, mean confidence %.3f",
        f"{applied:,}", f"{events_imputed:,}",
        f"{below_threshold:,}", summary['mean_confidence'],
    )
    return summary
