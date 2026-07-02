"""Feature extraction, leakage protection, and journey/session construction.

The unit of prediction is a GA4 *session* (channel attribution in
`session_traffic_source_last_click` is session-scoped), while behavioural
context comes from the user's whole journey.
"""

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Keys that directly reveal marketing attribution or act as unique
# identifiers which would cause the model to "cheat" rather than learn
# behaviour.  Matched exactly (lower-cased).
LEAKY_KEYS = {
    'campaign', 'medium', 'source', 'term', 'content',
    'gclid', 'fbclid', 'dclid', 'gclsrc', 'wbraid', 'gbraid', 'srsltid',
    'msclkid', 'ttclid', 'twclid', 'li_fat_id', 'igshid', 'epik', 'sccid',
    'manual_campaign_id', 'manual_campaign_name',
    'traffic_source_name', 'traffic_source_medium', 'traffic_source_source',
    'click_id', 'session_id',
    # High-cardinality identifiers: these are unique(ish) per session/user,
    # so the model would memorise them instead of learning behaviour.
    'ga_session_id', 'ga_session_number', 'firebase_screen_id',
}

# Substrings that indicate attribution/click-id params (utm_source,
# utm_campaign, gclid-style *clid params, custom click_id variants, ...).
LEAKY_KEY_SUBSTRINGS = ('utm_', 'click_id', 'clid')

# Event names strongly correlated with specific channels — bucket these
# to prevent soft leakage while retaining behavioural signal.
CHANNEL_CORRELATED_EVENTS = {
    'ad_click', 'ad_impression', 'ad_reward', 'ad_query',
    'campaign_details', 'dynamic_link_app_open', 'dynamic_link_first_open',
    'firebase_campaign', 'os_update', 'app_store_refund',
    'app_store_subscription_cancel', 'app_store_subscription_convert',
    'app_store_subscription_renew',
}

_MISSING_CHANNEL_VALUES = {
    "(none)", "", "unknown", "undefined", "null", "unassigned", "nan",
}


def is_missing(channel) -> bool:
    """Return True if the channel value is absent or represents a missing state."""
    if channel is None:
        return True
    if isinstance(channel, float) and math.isnan(channel):
        return True
    return str(channel).strip().lower() in _MISSING_CHANNEL_VALUES


def bucket_numeric(value, bins=(0, 1, 10, 100, 1000, 10000)) -> str:
    """Bucket a numeric value into a categorical string."""
    if value is None:
        return "unknown"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "unknown"
    if math.isnan(value):
        return "unknown"
    for b in bins:
        if value <= b:
            return f"<={b}"
    return f">{bins[-1]}"


def _safe_params_list(raw_params) -> list:
    """
    Normalise the all_params field from BigQuery into a list of dicts.
    Handles None, list-of-Row, list-of-dict, and other representations
    that vary across google-cloud-bigquery / pyarrow versions.
    """
    if raw_params is None or isinstance(raw_params, float):
        return []
    result = []
    try:
        for p in raw_params:
            if isinstance(p, dict):
                result.append(p)
            else:
                # google.cloud.bigquery Row or StructScalar — convert via dict()
                result.append(dict(p))
    except (TypeError, ValueError):
        return []
    return result


def _is_leaky_param(key: str) -> bool:
    if key in LEAKY_KEYS:
        return True
    return any(sub in key for sub in LEAKY_KEY_SUBSTRINGS)


def extract_event_features(e: dict) -> Counter:
    """
    Extract features from a single event dictionary.

    Mitigations applied:
      - event_name is bucketed for channel-correlated names to prevent soft
        leakage while retaining general behavioural signal.
      - Raw event_date replaced with day_of_week and hour_of_day for
        generalisable temporal signal (avoids unique-per-day features).
      - Attribution and identifier event params filtered via LEAKY_KEYS.
      - Integer param values are bucketed to bound cardinality (raw values
        like engagement_time_msec are near-unique and invite memorisation).
    """
    feat = Counter()

    # --- Event name (bucketed to prevent soft leakage) ---
    event_name = (e.get('event_name') or 'unknown').lower()
    if event_name in CHANNEL_CORRELATED_EVENTS:
        feat["event:__channel_correlated__"] += 1
    else:
        feat[f"event:{event_name}"] += 1

    # --- Temporal features (generalisable, not unique per date) ---
    event_ts = e.get('event_timestamp')
    if event_ts is not None:
        try:
            ts_sec = int(event_ts) / 1_000_000  # GA4 timestamps are microseconds
            dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            feat[f"day_of_week:{dt.weekday()}"] += 1
            feat[f"hour_of_day:{dt.hour}"] += 1
        except (ValueError, TypeError, OSError, OverflowError):
            feat["day_of_week:unknown"] += 1
            feat["hour_of_day:unknown"] += 1
    else:
        feat["day_of_week:unknown"] += 1
        feat["hour_of_day:unknown"] += 1

    # --- Event value ---
    feat[f"event_value_bucket:{bucket_numeric(e.get('event_value_in_usd'))}"] += 1

    # --- Device and Geo Context ---
    for feature_name, column in (
        ('device', 'device_category'),
        ('os', 'device_os'),
        ('mobile_brand', 'mobile_brand_name'),
        ('mobile_model', 'mobile_model_name'),
        ('os_version', 'operating_system_version'),
        ('language', 'language'),
        ('browser', 'browser'),
        ('browser_version', 'browser_version'),
        ('geo_country', 'geo_country'),
        ('geo_city', 'geo_city'),
        ('geo_continent', 'geo_continent'),
        ('geo_region', 'geo_region'),
        ('app_id', 'app_id'),
        ('app_version', 'app_version'),
        ('platform', 'platform'),
    ):
        value = e.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = 'unknown'
        feat[f"{feature_name}:{value}"] += 1

    # --- E-commerce context (behavioural, safe) ---
    feat[
        "purchase_revenue_bucket:"
        f"{bucket_numeric(e.get('purchase_revenue_in_usd'))}"
    ] += 1

    feat[f"active_user:{str(e.get('is_active_user'))}"] += 1

    # --- Custom event parameters (with leakage protection) ---
    for param in _safe_params_list(e.get('all_params')):
        key = str(param.get('param_key', '')).lower()
        if _is_leaky_param(key):
            continue

        val_str = param.get('param_str_val')
        val_int = param.get('param_int_val')

        if val_str is not None:
            feat[f"param:{key}:{val_str}"] += 1
        elif val_int is not None:
            # Bucket integers: raw values are often high-cardinality.
            feat[f"param:{key}:{bucket_numeric(val_int)}"] += 1

    return feat


# -----------------------------------------------------------------------------
# Journeys and sessions
# -----------------------------------------------------------------------------
def build_user_journeys(df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Group events by user into chronologically ordered journeys.

    Single vectorised sort + one linear pass — no per-user worker processes,
    which at scale cost more in serialisation than they saved in compute.
    Rows without a user_pseudo_id (common for unconsented traffic) each
    become a singleton journey rather than being merged into one fake user.
    """
    logger.info("Building user journeys from %s events", f"{len(df):,}")

    uid_missing = df['user_pseudo_id'].isna()
    if uid_missing.any():
        n = int(uid_missing.sum())
        df.loc[uid_missing, 'user_pseudo_id'] = [
            f"_anon_{i}" for i in range(n)
        ]
        logger.info("Assigned synthetic ids to %s events with no user_pseudo_id",
                    f"{n:,}")

    df.sort_values(['user_pseudo_id', 'event_timestamp'],
                   inplace=True, kind='stable')

    journeys: Dict[str, List[dict]] = {}
    for rec in df.to_dict('records'):
        rec['original_channel'] = rec.get('channel')
        rec['final_channel'] = rec.get('channel')
        rec['was_imputed'] = False
        rec['imputation_confidence'] = None
        journeys.setdefault(rec['user_pseudo_id'], []).append(rec)

    logger.info("Constructed journeys for %s users", f"{len(journeys):,}")
    return journeys


def group_sessions(events: List[dict]) -> List[dict]:
    """
    Group a user's chronologically ordered events into sessions by
    ga_session_id.  Events without a session id become singleton sessions
    (they cannot be reliably tied to any other ping).

    Returns a list of {'session_key': str, 'event_indices': [int, ...]}
    in order of first appearance.
    """
    sessions: List[dict] = []
    by_key: Dict[str, dict] = {}
    for i, e in enumerate(events):
        sid = e.get('ga_session_id')
        if sid is None or (isinstance(sid, float) and math.isnan(sid)):
            key = f"_nosid_{i}"
        else:
            key = str(int(sid)) if isinstance(sid, float) else str(sid)
        sess = by_key.get(key)
        if sess is None:
            sess = {'session_key': key, 'event_indices': []}
            by_key[key] = sess
            sessions.append(sess)
        sess['event_indices'].append(i)
    return sessions


def session_channel(events: List[dict], event_indices: List[int]) -> Optional[str]:
    """
    The known channel for a session, or None if unattributed.

    GA4 channel attribution is session-scoped so all events in a session
    should agree; if the export contains conflicts, take the majority label.
    """
    votes = Counter()
    for i in event_indices:
        ch = events[i].get('original_channel')
        if not is_missing(ch):
            votes[str(ch)] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]
