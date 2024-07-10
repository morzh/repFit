from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DPATH = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"

RESULTS_DPATH = RESULTS_ROOT / "classifier"


for path in [RESULTS_DPATH]:
    path.mkdir(exist_ok=True, parents=True)

