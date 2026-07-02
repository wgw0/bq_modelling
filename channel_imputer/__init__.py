"""Channel imputation for unconsented GA4 sessions.

Trains a dual-input deep learning model on sessions with known channel
attribution and imputes the most likely channel (with a confidence score)
for sessions where attribution is missing.
"""

__version__ = "2.0.0"
