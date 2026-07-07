"""
Configuration settings for CoLabScience model.
Architecture: Coordinator (6-layer MLP) + Observer (LLM) + Presenter (LLM) + Linear Projector
"""

import torch

# ============================================================================
# File paths
# ============================================================================
train_data_path = "../data/train.json"
val_data_path = "../data/validation.json"
test_data_path = "../data/test.json"

# Model save paths
observer_save_path = "saved_models/observer"
presenter_save_path = "saved_models/presenter"
presenter_sft_save_path = "./saved_models/presenter_sft"
coordinator_save_path = "saved_models/coordinator"
projector_save_path = "saved_models/projector"

# Set default data paths for SFT and GRPO
sft_train_path = "../data/sft_augmented_data.json"
llamafactory_sft_data_path = "data/presenter_sft_data.json"
grpo_train_path = "../data/grpo_pn_data.json"

# ============================================================================
# Observer LLM Configuration
# ============================================================================
observer_model_name = "meta-llama/Llama-3.2-1B-Instruct"
observer_hidden_dim = 2048  # Llama-3.2-1B hidden dimension
observer_tensor_parallel_size = 1
observer_learning_rate = 1e-6  # GRPO learning rate as per paper
observer_batch_size = 8
observer_max_length = 4096
observer_weight_decay = 1e-5  # AdamW weight decay
observer_warmup_ratio = 0.1  # 10% linear warm-up steps

# ============================================================================
# Presenter LLM Configuration  
# ============================================================================
presenter_model_name = "meta-llama/Llama-3.1-8B-Instruct"
presenter_hidden_dim = 4096  # Llama-3.1-8B hidden dimension
presenter_tensor_parallel_size = 2
presenter_learning_rate = 1e-5
presenter_batch_size = 8
presenter_max_length = 4096
presenter_max_output_length = 512  # Max output tokens

# LoRA Configuration for Presenter SFT
lora_rank = 16
lora_alpha = 64
lora_dropout = 0.1

# ============================================================================
# Linear Projector Configuration
# ============================================================================
projector_input_dim = presenter_hidden_dim  # 4096
projector_output_dim = observer_hidden_dim  # 2048 (project to observer dimension)
projector_learning_rate = 1e-3

# ============================================================================
# Coordinator MLP Configuration (6 layers)
# ============================================================================
coordinator_input_dim = observer_hidden_dim + projector_output_dim  # 4096 (2048 + 2048)
coordinator_hidden_dims = [4096, 2048, 1024, 512, 256, 128]  # 6 hidden layers
coordinator_output_dim = 1  # Binary classification (positive probability)
coordinator_dropout = 0.1
coordinator_learning_rate = 1e-4  # 6-layer MLP learning rate
coordinator_epochs = 50
coordinator_batch_size = 32

# ============================================================================
# Training Configuration
# ============================================================================
# Action threshold for positive/negative classification
action_threshold = 0.5  # >= 0.5 for potential positive, < 0.5 for potential negative

# Reward weighting
reward_weight_lambda = 0.6  # Balance between Observer (λ) and Presenter (1-λ) rewards

# Overall training epochs
total_training_epochs = 30
warmup_epochs = 5  # Initial epochs for individual component training

# ============================================================================
# SFT (Supervised Fine-Tuning) Configuration
# ============================================================================
sft_epochs = 3
sft_learning_rate = 1e-5
sft_batch_size = 4
sft_gradient_accumulation_steps = 8
sft_max_seq_length = 4096
sft_save_steps = 500

# ============================================================================
# GRPO Configuration
# ============================================================================
grpo_epochs = 3
grpo_learning_rate = 1e-6  # GRPO learning rate (AdamW optimizer)
grpo_batch_size = 4
grpo_vllm_server_host = "127.0.0.1"  # Localhost for GRPO vLLM server
grpo_num_processes = 5
grpo_cuda_devices = "3,4,5,6,7"
grpo_log_file_path = "train_grpo_model.log"
grpo_max_seq_length = 4096
grpo_save_steps = 500

# ============================================================================
# Evaluation Configuration
# ============================================================================
# Evaluation batch size
eval_batch_size = 16

# ============================================================================
# Device and Hardware Configuration
# ============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# ============================================================================
# Prompt Templates
# ============================================================================
# System prompt for both Observer and Presenter
system_prompt_template = """
You are a helpful AI assistant specializing in scientific research collaboration.
Your task is to understand and generate scientific interventions based on research contexts.
"""

# User prompt template for input processing
user_prompt_template = """
Project Context: {project_context}

Long-term Memory: {long_term_memory}

Recent Conversation:
{recent_conversation}

Task: {task_instruction}
"""

# Task instructions for different purposes
observer_task_instruction = """
Analyze the above conversation and context. Your role is to observe and understand the current state of the research discussion.
Provide your understanding and analysis of the situation.
"""

presenter_task_instruction = """
Based on the conversation and context above, provide a helpful scientific intervention that keeps the discussion aligned with the project goal. Focus on:
1. Redirecting if the conversation is going off-track
2. Introducing relevant information or connections  
3. Suggesting next steps to advance the research
4. Clarifying any scientific misconceptions

Your intervention:
"""

# Observer intervention judgment system prompt
observer_intervention_system_prompt = """
You are an AI moderator specializing in research coherence and integrity. Your task is to analyze a multi-turn scientific team discussion and determine whether the specified round of conversation requires an intervention.
Your response must be one of the following:
- Intervention Content: <brief reason>
- No Need Intervention
"""

# Observer intervention judgment user prompt template
observer_intervention_user_prompt = """
Project Context: {project_context}

Long-term Memory: {long_term_memory}

Recent Conversation:
{recent_conversation}

Please analyze whether this conversation requires an intervention to maintain research coherence and integrity.
"""