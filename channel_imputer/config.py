"""Pipeline configuration.

All tunables live in one immutable dataclass so a run is fully described by
its config (which is persisted alongside the model artifacts).  Values can be
overridden from the command line; secrets/credentials resolve via the
service-account file if present, otherwise Application Default Credentials.
"""

import argparse
import dataclasses
import re
from dataclasses import dataclass
from typing import Optional

DEFAULT_BQ_TABLE = "gtm-5vcmpn9-ntzmm.analytics_345697125.events_*"

# project.dataset.table — table part may end in a wildcard for sharded GA4
# exports.  The table id is interpolated into SQL, so it must be validated.
_TABLE_ID_RE = re.compile(
    r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+\*?$"
)
_DATE_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class PipelineConfig:
    # --- Data source ---
    bq_table: str = DEFAULT_BQ_TABLE
    date_start: str = "20250701"
    date_end: str = "20260131"
    service_account_file: str = "service-account-key.json"
    # Hard cost cap for the BigQuery job (bytes). None = no cap.
    max_bytes_billed: Optional[int] = None
    # Optional LIMIT on the query — cheap smoke runs against live data.
    row_limit: Optional[int] = None
    bq_page_size: int = 100_000

    # --- Features / model ---
    max_seq_length: int = 20
    feature_dim: int = 4096
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 0.001
    random_state: int = 12
    time_split_fraction: float = 0.8

    # --- Inference ---
    inference_batch_size: int = 512
    # Sessions predicted with max probability below this threshold are left
    # unimputed (exported with their original missing channel) rather than
    # being assigned a low-confidence guess.  0.0 imputes everything.
    confidence_threshold: float = 0.0

    # --- Outputs ---
    artifact_root: str = "models"
    # Optional BigQuery destination for the imputed events
    # (project.dataset.table, WRITE_TRUNCATE).
    output_bq_table: Optional[str] = None

    def validate(self) -> None:
        if not _TABLE_ID_RE.match(self.bq_table):
            raise ValueError(
                f"Invalid BigQuery table id: {self.bq_table!r} "
                "(expected project.dataset.table, optionally ending in *)"
            )
        if self.output_bq_table and not _TABLE_ID_RE.match(self.output_bq_table):
            raise ValueError(
                f"Invalid output table id: {self.output_bq_table!r}"
            )
        for label, value in (("date_start", self.date_start),
                             ("date_end", self.date_end)):
            if not _DATE_RE.match(value):
                raise ValueError(f"{label} must be YYYYMMDD, got {value!r}")
        if self.date_start > self.date_end:
            raise ValueError("date_start must be <= date_end")
        if not 0.0 < self.time_split_fraction < 1.0:
            raise ValueError("time_split_fraction must be in (0, 1)")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_seq_length < 1 or self.feature_dim < 2:
            raise ValueError("max_seq_length and feature_dim must be positive")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prop_model_app",
        description=(
            "Train a channel-imputation model on GA4 BigQuery event data and "
            "impute channels for unconsented (unattributed) sessions."
        ),
    )
    d = PipelineConfig()
    p.add_argument("--table", default=d.bq_table,
                   help="GA4 events table (project.dataset.events_*)")
    p.add_argument("--start", default=d.date_start, metavar="YYYYMMDD",
                   help="First event date (inclusive)")
    p.add_argument("--end", default=d.date_end, metavar="YYYYMMDD",
                   help="Last event date (inclusive)")
    p.add_argument("--service-account-file", default=d.service_account_file,
                   help="Path to a service-account key JSON; falls back to "
                        "Application Default Credentials if the file is absent")
    p.add_argument("--max-gb-billed", type=float, default=None,
                   help="Fail the BigQuery job if it would bill more than "
                        "this many GB (cost guard)")
    p.add_argument("--limit", type=int, default=None,
                   help="LIMIT for the BigQuery query (smoke runs)")
    p.add_argument("--max-seq-length", type=int, default=d.max_seq_length)
    p.add_argument("--feature-dim", type=int, default=d.feature_dim,
                   help="Hashed feature dimension (memory/quality trade-off)")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--random-state", type=int, default=d.random_state)
    p.add_argument("--time-split-fraction", type=float,
                   default=d.time_split_fraction)
    p.add_argument("--confidence-threshold", type=float,
                   default=d.confidence_threshold,
                   help="Leave sessions unimputed when the model's confidence "
                        "is below this value (0 imputes everything)")
    p.add_argument("--artifact-root", default=d.artifact_root)
    p.add_argument("--export-bq-table", default=None, metavar="TABLE",
                   help="Also write imputed events to this BigQuery table "
                        "(project.dataset.table, replaces existing contents)")
    return p


def config_from_args(argv=None) -> PipelineConfig:
    args = _build_parser().parse_args(argv)
    max_bytes = (
        int(args.max_gb_billed * 1024 ** 3)
        if args.max_gb_billed is not None else None
    )
    cfg = PipelineConfig(
        bq_table=args.table,
        date_start=args.start,
        date_end=args.end,
        service_account_file=args.service_account_file,
        max_bytes_billed=max_bytes,
        row_limit=args.limit,
        max_seq_length=args.max_seq_length,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        time_split_fraction=args.time_split_fraction,
        confidence_threshold=args.confidence_threshold,
        artifact_root=args.artifact_root,
        output_bq_table=args.export_bq_table,
    )
    cfg.validate()
    return cfg
