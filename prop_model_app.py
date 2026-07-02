#!/usr/bin/env python
"""Channel imputation for unconsented GA4 sessions — CLI entrypoint.

Trains a dual-input deep learning model on sessions with known channel
attribution and imputes the most likely channel (with a confidence score)
for sessions where attribution is missing.

Run `python prop_model_app.py --help` for all options.  The implementation
lives in the `channel_imputer` package.
"""

import logging
import sys

from channel_imputer.config import config_from_args


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    try:
        cfg = config_from_args(argv)
    except ValueError as exc:
        logging.getLogger(__name__).error("Invalid configuration: %s", exc)
        return 2

    # Imported here so --help and config errors don't pay the TensorFlow
    # import cost.
    from channel_imputer.pipeline import run
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
