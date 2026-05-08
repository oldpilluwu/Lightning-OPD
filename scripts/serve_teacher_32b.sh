# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --host 127.0.0.1 \
    --port 13141 \
    --tp 8 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.6 \
    --context-length 8192 \
    > "$LOG_FILE" 2>&1 &

until curl -sf http://127.0.0.1:13141/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    sleep 5
done