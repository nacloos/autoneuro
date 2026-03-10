"""Fixed experiment configuration for autoneuro."""

from pathlib import Path

# Fixed target setup (modular-only, multitask Yang)
TASK_SET = "all_yang"
MODEL = "modular"
K = "1,1"
N_TRAIN_SAMPLES = 500  # per task
N_TEST_SAMPLES = 500   # per task
N_LAYERS = 2
SHARED_WEIGHTS = True
CONCAT_INPUT = True
BATCH_SIZE = 64
MAX_TRAIN_STEPS = 5000
N_EPOCHS = 200
SEED = 0
NO_WANDB = True

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
