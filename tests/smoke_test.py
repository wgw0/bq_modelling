"""End-to-end smoke test on synthetic data — no BigQuery access required.

Generates user journeys whose channel correlates with device/geo/time
signals, runs the full pipeline (features → train → evaluate → impute →
export), and asserts the outputs are well-formed.  Run with:

    python tests/smoke_test.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_imputer.config import PipelineConfig  # noqa: E402
from channel_imputer.pipeline import run  # noqa: E402

CHANNELS = {
    'Organic Search': {'device': 'desktop', 'country': 'United Kingdom',
                       'event': 'page_view', 'hour': 10},
    'Paid Social':    {'device': 'mobile', 'country': 'United States',
                       'event': 'ad_click', 'hour': 20},
    'Email':          {'device': 'tablet', 'country': 'Germany',
                       'event': 'select_promotion', 'hour': 7},
}

BASE_TS = 1_760_000_000_000_000  # microseconds
DAY_US = 86_400_000_000


def make_events(n_users=120, sessions_per_user=3, events_per_session=5,
                missing_fraction=0.3, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    channel_names = list(CHANNELS)
    for u in range(n_users):
        user = f"user_{u}"
        channel = channel_names[u % len(channel_names)]
        profile = CHANNELS[channel]
        for s in range(sessions_per_user):
            session_id = u * 100 + s
            missing = rng.random() < missing_fraction
            start = BASE_TS + int(rng.integers(0, 90)) * DAY_US
            for k in range(events_per_session):
                # a little noise so classes are not perfectly separable
                noisy = rng.random() < 0.1
                rows.append({
                    'event_timestamp': start + k * 60_000_000
                    + profile['hour'] * 3_600_000_000,
                    'event_name': profile['event'] if not noisy else 'scroll',
                    'event_value_in_usd': float(rng.random() * 50),
                    'user_pseudo_id': user,
                    'ga_session_id': session_id,
                    'device_category': profile['device'],
                    'device_os': 'android' if profile['device'] == 'mobile'
                                 else 'windows',
                    'mobile_brand_name': None,
                    'mobile_model_name': None,
                    'operating_system_version': '11',
                    'language': 'en-gb',
                    'browser': 'chrome',
                    'browser_version': '120',
                    'geo_country': profile['country'],
                    'geo_city': 'City',
                    'geo_continent': 'Europe',
                    'geo_region': 'Region',
                    'app_id': None,
                    'app_version': None,
                    'platform': 'WEB',
                    'purchase_revenue_in_usd': None,
                    'is_active_user': True,
                    'channel': None if missing else channel,
                    'all_params': [
                        {'param_key': 'ga_session_id', 'param_int_val': session_id,
                         'param_str_val': None},
                        {'param_key': 'page_title', 'param_str_val': 'Home',
                         'param_int_val': None},
                        {'param_key': 'engagement_time_msec',
                         'param_int_val': int(rng.integers(100, 60_000)),
                         'param_str_val': None},
                    ],
                })
    # a few fully-anonymous events with no user or session id
    for i in range(10):
        rows.append({
            'event_timestamp': BASE_TS + i * DAY_US,
            'event_name': 'page_view',
            'event_value_in_usd': None,
            'user_pseudo_id': None,
            'ga_session_id': None,
            'device_category': 'mobile',
            'device_os': 'ios',
            'mobile_brand_name': 'Apple',
            'mobile_model_name': 'iPhone',
            'operating_system_version': '17',
            'language': 'en-us',
            'browser': 'safari',
            'browser_version': '17',
            'geo_country': 'United States',
            'geo_city': 'City',
            'geo_continent': 'Americas',
            'geo_region': 'Region',
            'app_id': None,
            'app_version': None,
            'platform': 'WEB',
            'purchase_revenue_in_usd': None,
            'is_active_user': None,
            'channel': None,
            'all_params': None,
        })
    return pd.DataFrame(rows)


def main():
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    artifact_root = tempfile.mkdtemp(prefix="channel_imputer_smoke_")
    cfg = PipelineConfig(
        feature_dim=256,
        max_seq_length=6,
        epochs=4,
        batch_size=32,
        artifact_root=artifact_root,
        confidence_threshold=0.0,
    )
    try:
        report = run(cfg, events_df=make_events())

        # --- Assertions -------------------------------------------------
        assert report['train_sessions'] > 0
        assert report['val_time_sessions'] > 0
        assert report['imputation']['sessions_imputed'] > 0
        assert report['imputation']['events_imputed'] > 0
        assert 0.0 <= report['val_time_accuracy'] <= 1.0

        run_dir = os.path.join(artifact_root, report['run_id'])
        for artifact in ("model.h5", "feature_hasher.joblib",
                         "label_encoder.joblib", "run_config.json",
                         "evaluation.json", "history.json",
                         "imputed_events.csv"):
            path = os.path.join(run_dir, artifact)
            assert os.path.exists(path), f"missing artifact: {artifact}"

        out = pd.read_csv(os.path.join(run_dir, "imputed_events.csv"))
        assert set(out.columns) == {
            'user_pseudo_id', 'ga_session_id', 'event_timestamp',
            'event_name', 'original_channel', 'final_channel',
            'was_imputed', 'imputation_confidence',
        }
        imputed = out[out['was_imputed'] == True]  # noqa: E712
        assert len(imputed) > 0
        assert imputed['final_channel'].notna().all()
        assert imputed['imputation_confidence'].between(0, 1).all()
        # imputed channels must come from the training label space
        assert set(imputed['final_channel']).issubset(set(CHANNELS))
        # sessions must be internally consistent after imputation
        by_session = imputed.dropna(subset=['ga_session_id']).groupby(
            ['user_pseudo_id', 'ga_session_id'])['final_channel'].nunique()
        assert (by_session == 1).all(), "conflicting channels within a session"
        # known labels must never be overwritten
        known = out[out['original_channel'].notna()]
        assert (known['original_channel'] == known['final_channel']).all()

        # sanity: synthetic classes are learnable, expect decent accuracy
        print(f"\nval_time_accuracy: {report['val_time_accuracy']:.3f}")
        print(f"imputed sessions:  {report['imputation']['sessions_imputed']}")
        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
