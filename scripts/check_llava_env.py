"""Verify the exact software stack used by the LLaVA-v1.5 baseline."""

import accelerate
import torch
import torchvision
import transformers

import llava


def main() -> None:
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("transformers:", transformers.__version__)
    print("accelerate:", accelerate.__version__)
    print("llava module:", llava.__file__)
    print("CUDA build:", torch.version.cuda)


if __name__ == "__main__":
    main()
