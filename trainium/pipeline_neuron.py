# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint for the native PyTorch curation pipeline.

New code should invoke ``python -m trainium.sft_data_generation_native.pipeline``.
"""

from trainium.sft_data_generation_native.pipeline import main


if __name__ == "__main__":
    main()
