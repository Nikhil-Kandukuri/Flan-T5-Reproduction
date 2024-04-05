"""
Download the subsets of the Flan Collection

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import traceback
from datasets import load_dataset

SUBMIX_NAMES = [
    "DataProvenanceInitiative/t0_submix_original",
    "DataProvenanceInitiative/dialog_submix_original",
    "DataProvenanceInitiative/niv2_submix_original",
    "DataProvenanceInitiative/cot_submix_original",
    "DataProvenanceInitiative/flan2021_submix_original",
]


def main(args):
    logging.info("Download the Flan Collection")

    for sub_mix in SUBMIX_NAMES:
        logging.info(f"Download: {sub_mix}")
        try:
            load_dataset(sub_mix)
            logging.info("Download completed")
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser(description="Download the Flan Collection")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    main(args)
