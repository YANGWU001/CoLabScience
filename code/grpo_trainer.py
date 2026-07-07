import argparse
import os
import json
import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import re
from config import system_prompt_template, user_prompt_template, observer_model_name, observer_intervention_system_prompt, observer_intervention_user_prompt
from peft import LoraConfig, TaskType
import torch.distributed as dist





def convert_to_prompt_completion_dataset(train):
    processed_data = []
    for item in train:
        # Check if this is the new format with Observer intervention judgment
        if "ground_truth" in item and isinstance(item["ground_truth"], int):
            # New format: Observer intervention judgment
            processed_data.append({
                "prompt": item.get("prompt", ""),
                "completion": item.get("completion", ""),
                "ground_truth": item.get("ground_truth", 0)  # 0 or 1 for intervention judgment
            })
        else:
            # Old format: Presenter intervention generation (backward compatibility)
            proposal = item.get("proposal", "").strip()
            ltm = item.get("summarized_long_term_memory", "").strip()
            stm_list = item.get("short_term_memory", [])
            stm = "\n".join(stm_list).strip()
            completion = item.get("Intervention content", "").strip()
            prompt = system_prompt_template + user_prompt_template.format(project_context=proposal,long_term_memory=ltm,recent_conversation=stm, task_instruction="Generate intervention content")
            processed_data.append({
                "prompt": prompt,
                "completion": completion,
                "ground_truth": completion
            })
    return Dataset.from_list(processed_data)



def reward_intervention_judgment(completions, ground_truth_labels, **kwargs):
    """
    Reward function for Observer intervention judgment.
    Compares model predictions with ground truth intervention labels.
    
    Args:
        completions: List of model responses 
        ground_truth_labels: List of ground truth labels (0=no intervention, 1=intervention needed)
        
    Returns:
        List of rewards (1.0 for correct prediction, 0.0 for incorrect)
    """
    if ground_truth_labels is None:
        raise ValueError("Expected 'ground_truth_labels' for intervention judgment.")
    
    rewards = []
    for completion, ground_truth in zip(completions, ground_truth_labels):
        # Parse model response to determine prediction
        prediction = parse_intervention_response(completion)
        
        # Reward: 1.0 if prediction matches ground truth, 0.0 otherwise
        reward = 1.0 if prediction == ground_truth else 0.0
        rewards.append(reward)
        
        print(f"[DEBUG] Completion: '{completion.strip()[:50]}...' | GT: {ground_truth} | Pred: {prediction} | Reward: {reward}")
    
    print(f"[DEBUG] Intervention Judgment Rewards: {rewards}")
    return rewards


def parse_intervention_response(response):
    """
    Parse model response to determine intervention judgment prediction.
    
    Args:
        response: Model response string
        
    Returns:
        int: 0 if no intervention needed, 1 if intervention needed
    """
    response_lower = response.lower().strip()
    
    # Check for "no need intervention" patterns (case insensitive)
    if "no need" in response_lower:
        return 0  # No intervention needed
    elif "intervention content:" in response_lower:
        return 1  # Intervention needed
    elif "intervention" in response_lower and "no" not in response_lower:
        return 1  # Likely intervention needed
    else:
        # Default: if unclear, assume no intervention
        return 0


# Keep the old function for backward compatibility
def reward_simple_quality(completions, ground_truth, **kwargs):
    """
    Simple quality-based reward function (for backward compatibility).
    """
    if ground_truth is None:
        raise ValueError("Expected 'completion' as ground-truth reference.")
    
    # Simple reward based on completion length and basic quality checks
    rewards = []
    for completion in completions:
        # Basic quality score based on length and content
        score = min(len(completion.strip()) / 100.0, 1.0)  # Normalize by length
        if len(completion.strip()) < 10:  # Too short
            score = 0.1
        rewards.append(score)
    
    print(f"[DEBUG] Simple Quality Scores: {rewards}")
    return rewards



def get_last_checkpoint(output_dir, base_llm_path):
    """
    Get the most recent checkpoint based on modification time.
    """
    checkpoints = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
    ]
    if not checkpoints:
        print(f"[Warning] No checkpoints found in {output_dir}.")
        return base_llm_path

    # Get path and modification time for each checkpoint
    checkpoints_with_time = [
        (os.path.join(output_dir, d), os.path.getmtime(os.path.join(output_dir, d)))
        for d in checkpoints
    ]

    # Sort by modification time and select the latest checkpoint
    latest_checkpoint = max(checkpoints_with_time, key=lambda x: x[1])[0]
    print(f"[INFO] Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[INFO] Set CUDA device to local_rank: {local_rank}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--vllm_server_host", type=str, default="", help="e.g., xx.xxx.xxx.xxx")
    parser.add_argument("--output_dir", type=str, default="GRPO", help="Checkpoint and model output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.train_path, "r", encoding="utf-8") as f:
        train = json.load(f)


    last_ckpt = get_last_checkpoint(args.output_dir, observer_model_name)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        bf16=True,
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        resume_from_checkpoint=last_ckpt if last_ckpt else None,
        use_vllm=True,
        max_completion_length=256,
        max_prompt_length=4096,
        deepspeed="./accelerate_configs/deepspeed_zero3.json",
        vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),
        vllm_server_port=8003,
        learning_rate= 1e-04
    )
    print(f"[INFO] Resume from: {last_ckpt}" if last_ckpt else "[INFO] Training from scratch")

    trainer = GRPOTrainer(
        model = observer_model_name,
        args=training_args,
        reward_funcs=reward_intervention_judgment,
        train_dataset=convert_to_prompt_completion_dataset(train),
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        if dist.is_initialized():
            print("[INFO] Destroying torch.distributed process group to clean up NCCL resources.")
            dist.destroy_process_group()

if __name__ == "__main__":
    main()


