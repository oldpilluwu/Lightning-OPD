# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

docker run -it --gpus all \
    --shm-size=64g \
    -v $(pwd):/workspace/Lightning-OPD \
    -v $HOME/.cache:$HOME/.cache \
    -w /workspace/Lightning-OPD \
    tonyhao96/jetmoe:v0.2 \
    bash
