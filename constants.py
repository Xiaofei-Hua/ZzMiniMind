import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR = os.path.join(BASE_DIR, "out")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

PRETRAIN_T2T_DATASET_DIR = os.path.join(DATASET_DIR, "pretrain_t2t.jsonl")
PRETRAIN_T2T_MINI_DATASET_DIR = os.path.join(DATASET_DIR, "pretrain_t2t_mini.jsonl")


if __name__ == "__main__":
    print(OUT_DIR)