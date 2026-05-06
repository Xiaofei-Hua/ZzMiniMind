#!/usr/bin/env bash
set -u

# ===== 基本配置 =====
SESSION_NAME="pretrain"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# ===== 如果不在 tmux 里, 就自动进入 tmux 再运行本脚本 =====
if [ -z "${TMUX:-}" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux 未安装, 请先执行: "
        echo "sudo apt update && sudo apt install tmux"
        exit 1
    fi

    echo "当前不在 tmux 中, 正在创建/进入 tmux 会话: ${SESSION_NAME}"

    # 如果会话已存在, 就直接 attach
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "tmux 会话已存在, 正在进入: ${SESSION_NAME}"
        tmux attach -t "${SESSION_NAME}"
        exit 0
    fi

    # 创建新 tmux 会话，并在里面运行本脚本
    # 训练脚本退出后保留窗口，方便查看错误信息
    tmux new-session -s "${SESSION_NAME}" "bash '${BASH_SOURCE[0]}'; echo; echo '脚本已退出，按 Enter 关闭 tmux 窗口'; read"
    exit 0
fi

# ===== 以下内容只会在 tmux 内执行 =====
LOG_FILE="${LOG_DIR}/pretrain_$(date '+%Y%m%d_%H%M%S').log"

cd "${SCRIPT_DIR}" || exit 1

{
    echo "========== Training Started =========="
    echo "Start time: $(date)"
    echo "Script dir: ${SCRIPT_DIR}"
    echo "Current dir: $(pwd)"
    echo "Log file: ${LOG_FILE}"
    echo "Shell PID: $$"
    echo "tmux session: ${SESSION_NAME}"
    echo "======================================"
    echo

    echo "========== Training Config =========="
    echo "use_moe: 0"
    echo "batch_size: 48"
    echo "accumulation_steps: 8"
    echo "effective_batch_size: 256"
    echo "max_seq_len: 512"
    echo "learning_rate: 5e-4"
    echo "epochs: 1"
    echo "num_workers: 4"
    echo "dtype: bfloat16"
    echo "grad_clip: 1.0"
    echo "log_interval: 500"
    echo "====================================="
    echo

    PYTHONUNBUFFERED=1 python -u trainer/train_full_sft.py \
        --use_moe 0 \
        --batch_size 48 \
        --accumulation_steps 8 \
        --max_seq_len 512 \
        --learning_rate 5e-4 \
        --epochs 1 \
        --num_workers 4 \
        --dtype bfloat16 \
        --grad_clip 1.0 \
        --log_interval 500

    EXIT_CODE=$?

    echo
    echo "========== Training Finished =========="
    echo "End time: $(date)"
    echo "Exit code: ${EXIT_CODE}"
    echo "Log file: ${LOG_FILE}"
    echo "======================================="

    exit "${EXIT_CODE}"
} 2>&1 | tee -a "${LOG_FILE}"