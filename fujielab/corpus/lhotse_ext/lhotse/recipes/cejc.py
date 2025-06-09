from typing import Optional, Union
from lhotse.utils import Pathlike


# Evaluation Speaker IDs
_EVAL = [
   "T015_021", "T018_003",             # male age 10-
   "K002_003", "T018_004",             # female age 10-
   "C002_002", "T006_009", "T010_002", # male age 20-
   "K001_018", "S002_002", "T009_015", # female age 20-
   "K005_015", "T005_006", "W008_003", # male age 30-
   "C002_010", "K002_007", "K004_012", # female age 30-
   "K007_009", "T008_008", "T015_007", # male age 40-
   "K004_008", "T007_011", "T020_022", # female age 40-
   "K010_005", "T008_003", "T022_006", # male age 50-
   "K002_004", "T011_005", "T017_012", # female age 50-
   "K005_004", "T013_023", "T017_002", "T023_005", # male age 60-
   "K006_013", "T006_001", "T008_007", # female age 60-
   "T004_011", "T005_052",             # male age 70-
   "K009_020", "T014_002",             # female age 70-
]


def prepare_cejc(
    corpus_dir: str,
    corpus_media_data_dir: Optional[str] = None,
    transcript_dir: Pathlike = None,
    manifest_dir: Pathlike = None,
)