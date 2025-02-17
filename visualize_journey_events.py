import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google.oauth2 import service_account
from google.cloud import bigquery

def main():
    # =============================================================================
    # STEP 0: Set up BigQuery Client
    # =============================================================================
    print("=" * 30)
    print("Setting up BigQuery client using service-account-key.json")
    credentials = service_account.Credentials.from_service_account_file("service-account-key.json")
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    print("Client created for project:", credentials.project_id)
    print("=" * 30, "\n")
    
    # =============================================================================
    # STEP 1: Query Event-Level Data from BigQuery
    # =============================================================================
    print("=" * 30)
    print("Querying event-level data from BigQuery to calculate journey lengths")
    query = """
    SELECT
      user_pseudo_id,
      event_timestamp
    FROM `gtm-5vcmpn9-ntzmm.analytics_345697125.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '20241001' AND '20250228'
      AND event_timestamp IS NOT NULL
    ORDER BY user_pseudo_id, event_timestamp
    """
    df = client.query(query).to_dataframe()
    print(f"Data retrieved: {len(df)} rows.")
    print("=" * 30, "\n")
    
    # =============================================================================
    # STEP 2: Calculate Journey Lengths
    # =============================================================================
    print("=" * 30)
    print("Calculating journey lengths by grouping events per user")
    journey_counts = df.groupby('user_pseudo_id').size()  # Series of counts per user

    # Print descriptive statistics
    print("Journey Length Descriptive Statistics:")
    print(journey_counts.describe())
    
    # Calculate specific quantiles to understand distribution
    quantiles = journey_counts.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    print("\nJourney Length Quantiles:")
    print(quantiles)
    print("=" * 30, "\n")
    
    # =============================================================================
    # STEP 3: Plot 1 - Standard Histogram with Log Scale
    # =============================================================================
    print("=" * 30)
    print("Plot 1: Standard histogram (log scale on X-axis)")
    plt.figure(figsize=(10, 6))
    plt.hist(journey_counts, bins=50, color='skyblue', edgecolor='black')
    plt.xscale('log')  # log-scale on the x-axis
    plt.title('Distribution of Journey Event Counts (Log-Scaled X-axis)')
    plt.xlabel('Number of Events in Journey (log scale)')
    plt.ylabel('Frequency')
    
    mean_val = journey_counts.mean()
    median_val = journey_counts.median()
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("journey_length_log_scale.png")
    print("Plot saved as 'journey_length_log_scale.png'")
    plt.show()
    print("=" * 30, "\n")
    
    # =============================================================================
    # STEP 4: Plot 2 - Clipped Histogram (Below 95th Percentile)
    # =============================================================================
    clip_threshold = journey_counts.quantile(0.95)
    print("=" * 30)
    print(f"Plot 2: Histogram clipped at 95th percentile (~{clip_threshold:.1f} events)")
    journey_counts_clipped = journey_counts[journey_counts <= clip_threshold]
    
    plt.figure(figsize=(10, 6))
    plt.hist(journey_counts_clipped, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of Journey Event Counts (<= {clip_threshold:.0f})')
    plt.xlabel('Number of Events in Journey')
    plt.ylabel('Frequency')
    
    mean_val_clipped = journey_counts_clipped.mean()
    median_val_clipped = journey_counts_clipped.median()
    plt.axvline(mean_val_clipped, color='red', linestyle='dashed', linewidth=1.5, 
                label=f'Mean (clipped): {mean_val_clipped:.1f}')
    plt.axvline(median_val_clipped, color='green', linestyle='dashed', linewidth=1.5, 
                label=f'Median (clipped): {median_val_clipped:.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("journey_length_clipped.png")
    print("Plot saved as 'journey_length_clipped.png'")
    plt.show()
    print("=" * 30, "\n")
    
    # =============================================================================
    # STEP 5: Plot 3 - Boxplot
    # =============================================================================
    print("=" * 30)
    print("Plot 3: Boxplot of Journey Lengths (may show outliers)")
    plt.figure(figsize=(3, 8))
    plt.boxplot(journey_counts, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='skyblue'))
    plt.title("Boxplot of Journey Lengths")
    plt.ylabel("Number of Events in Journey")
    plt.savefig("journey_length_boxplot.png")
    print("Plot saved as 'journey_length_boxplot.png'")
    plt.show()
    print("=" * 30, "\n")

if __name__ == "__main__":
    main()
