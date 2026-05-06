import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR = os.path.join(BASE_DIR, "out")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# ModelScope 数据集缓存目录
DATASET_DIR = os.path.expanduser(
    "~/.cache/modelscope/hub/datasets/gongjy/minimind_dataset"
)

PRETRAIN_T2T_DATASET_PATH = os.path.join(DATASET_DIR, "pretrain_t2t.jsonl")
PRETRAIN_T2T_MINI_DATASET_PATH = os.path.join(DATASET_DIR, "pretrain_t2t_mini.jsonl")
SFT_T2T_MINI_DATASET_PATH = os.path.join(DATASET_DIR, "sft_t2t_mini.jsonl")
SFT_T2T_DATASET_PATH = os.path.join(DATASET_DIR, "sft_t2t.jsonl")
RLAIF_DATASET_PATH = os.path.join(DATASET_DIR, "RLAIF.jsonl")


# reward model 缓存目录
REWARD_PATH = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward"
)


if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATASET_DIR:", DATASET_DIR)
    print("PRETRAIN_T2T_DATASET_PATH:", PRETRAIN_T2T_DATASET_PATH)
    print("PRETRAIN_T2T_MINI_DATASET_PATH:", PRETRAIN_T2T_MINI_DATASET_PATH)
    print("SFT_T2T_MINI_DATASET_PATH:", SFT_T2T_MINI_DATASET_PATH)
    print("RLAIF_DATASET_PATH:", RLAIF_DATASET_PATH)
    print("REWARD_PATH:", REWARD_PATH)