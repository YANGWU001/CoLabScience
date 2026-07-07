
"""
CoLabScience Training Starter
Script to avoid torch.distributed issues
"""

import os
import sys

# Set environment variables to avoid distributed issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'

# Disable distributed-related auto-detection
os.environ['NCCL_DISABLE_CHECK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Starting CoLabScience Training - Non-distributed Mode")
print("=====================================================")
print(f"CUDA Devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
print(f"World Size: {os.environ.get('WORLD_SIZE', 'N/A')}")
print()

# Import and run training
if __name__ == "__main__":
    from train import main
    main()