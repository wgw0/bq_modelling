"""BigQuery I/O: authenticated client, event query, and result export."""

import logging
import os

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def make_client(cfg: PipelineConfig) -> bigquery.Client:
    """
    Build a BigQuery client.  Prefers the configured service-account key
    file; falls back to Application Default Credentials so the pipeline can
    run on GCP infrastructure (Cloud Run, Vertex, GCE) without a key file.
    """
    if cfg.service_account_file and os.path.exists(cfg.service_account_file):
        credentials = service_account.Credentials.from_service_account_file(
            cfg.service_account_file
        )
        client = bigquery.Client(
            credentials=credentials, project=credentials.project_id
        )
        logger.info("BigQuery client created from service-account key "
                    "(project: %s)", credentials.project_id)
    else:
        client = bigquery.Client()  # Application Default Credentials
        logger.info("BigQuery client created from Application Default "
                    "Credentials (project: %s)", client.project)
    return client


def _build_query(cfg: PipelineConfig) -> str:
    # Only columns that are actually consumed downstream are selected —
    # BigQuery bills by column scanned.  The date range is passed as query
    # parameters; the (validated) table id is the only interpolation.
    # No ORDER BY: global sorts of large event exports routinely exhaust
    # BigQuery resources, and the pipeline sorts locally anyway.
    limit_clause = f"\n    LIMIT {int(cfg.row_limit)}" if cfg.row_limit else ""
    return f"""
    SELECT
      event_timestamp,
      event_name,
      event_value_in_usd,
      user_pseudo_id,
      (SELECT value.int_value FROM UNNEST(event_params)
       WHERE key = 'ga_session_id' LIMIT 1) AS ga_session_id,
      device.category AS device_category,
      device.operating_system AS device_os,
      device.mobile_brand_name,
      device.mobile_model_name,
      device.operating_system_version,
      device.language AS language,
      device.browser AS browser,
      device.browser_version AS browser_version,
      geo.country AS geo_country,
      geo.city AS geo_city,
      geo.continent AS geo_continent,
      geo.region AS geo_region,
      app_info.id AS app_id,
      app_info.version AS app_version,
      platform,
      ecommerce.purchase_revenue_in_usd,
      is_active_user,
      session_traffic_source_last_click.cross_channel_campaign.primary_channel_group AS channel,
      ARRAY(
        SELECT AS STRUCT
          ep.key AS param_key,
          ep.value.string_value AS param_str_val,
          ep.value.int_value AS param_int_val
        FROM UNNEST(event_params) ep
      ) AS all_params
    FROM `{cfg.bq_table}`
    WHERE
      _TABLE_SUFFIX BETWEEN @date_start AND @date_end
      AND event_timestamp IS NOT NULL{limit_clause}
    """


def query_event_data(client: bigquery.Client, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Query event-level data, streaming results in pages to bound peak memory.
    Applies the maximum_bytes_billed cost guard when configured.
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("date_start", "STRING", cfg.date_start),
            bigquery.ScalarQueryParameter("date_end", "STRING", cfg.date_end),
        ],
        maximum_bytes_billed=cfg.max_bytes_billed,
        labels={"app": "channel-imputer"},
    )
    logger.info("Querying %s for %s..%s%s",
                cfg.bq_table, cfg.date_start, cfg.date_end,
                f" (LIMIT {cfg.row_limit})" if cfg.row_limit else "")
    if cfg.max_bytes_billed:
        logger.info("Cost guard: maximum_bytes_billed = %.1f GB",
                    cfg.max_bytes_billed / 1024 ** 3)

    query_job = client.query(_build_query(cfg), job_config=job_config)
    row_iterator = query_job.result(page_size=cfg.bq_page_size)  # blocks

    if query_job.total_bytes_processed is not None:
        logger.info("Query complete: %.1f MB processed. Streaming results...",
                    query_job.total_bytes_processed / 1024 / 1024)

    frames = []
    rows_so_far = 0
    for page_num, page in enumerate(row_iterator.pages, start=1):
        page_rows = [dict(row) for row in page]
        if page_rows:
            frames.append(pd.DataFrame(page_rows))
            rows_so_far += len(page_rows)
            if page_num % 5 == 0:
                logger.info("  Downloaded %s rows (%d pages)...",
                            f"{rows_so_far:,}", page_num)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("Data retrieved: %s rows", f"{len(df):,}")

    if df.empty:
        raise RuntimeError(
            "BigQuery returned 0 rows — check the date range and table path."
        )
    return df


def export_to_bigquery(client: bigquery.Client, df: pd.DataFrame,
                       table_id: str) -> None:
    """Load the imputed events into a BigQuery table (replaces contents)."""
    logger.info("Loading %s rows into BigQuery table %s",
                f"{len(df):,}", table_id)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # wait and raise on failure
    logger.info("BigQuery export complete: %s", table_id)
