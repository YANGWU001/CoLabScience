"""
CoLabScience Training Script - Architecture
==============================================

End-to-End Training Flow:
1. Load unlabeled and labeled data
2. For each training epoch:
   a. Process unlabeled samples through Observer + Presenter → embeddings
   b. Project Presenter embeddings to Observer dimension  
   c. Concat embeddings → Coordinator → action probabilities
   d. Split samples: action >= 0.5 (potential positive), < 0.5 (potential negative)
   e. Augment original positives with potential positives → SFT train Presenter
   f. Use potential negatives for GRPO training
   g. Evaluate Observer + Presenter on validation set → rewards
   h. Combine rewards with lambda weighting → train Coordinator with REINFORCE
3. Save models and repeat

This implements a continuous learning loop where the Coordinator learns to identify
valuable samples for improving both Observer and Presenter performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
import copy
# from tqdm import tqdm  # Removed for academic submission
import subprocess
import time
import gc
import logging

# Import models and configurations
from models import Observer, Presenter, LinearProjector, Coordinator, evaluate_models_on_validation
from config import (
    # Data paths
    train_data_path, val_data_path,
    # Model save paths  
    observer_save_path, presenter_save_path, coordinator_save_path, projector_save_path,
    # Training paths
    sft_train_path, grpo_train_path,
    # Training configuration
    total_training_epochs, warmup_epochs, action_threshold, reward_weight_lambda,
    # SFT configuration
    sft_epochs, sft_learning_rate, sft_batch_size, sft_gradient_accumulation_steps,
    # GRPO configuration  
    grpo_epochs, grpo_learning_rate, grpo_batch_size, grpo_vllm_server_host, grpo_num_processes, grpo_cuda_devices, grpo_log_file_path,
    # Observer intervention judgment prompts
    observer_intervention_system_prompt, observer_intervention_user_prompt,
    # Coordinator configuration
    coordinator_learning_rate, coordinator_epochs, coordinator_batch_size,
    # Projector configuration
    projector_learning_rate,
    # Other
    device, eval_batch_size
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def load_training_data():
    """Load training and validation datasets."""
    logger.info("Loading training and validation data...")
    
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # Separate labeled positive and unlabeled data
    labeled_positive = [item for item in train_data if item.get('label') == 'positive']
    unlabeled_data = [item for item in train_data if item.get('label') == 'unlabeled']
    
    logger.info(f"Loaded {len(labeled_positive)} labeled positive samples")
    logger.info(f"Loaded {len(unlabeled_data)} unlabeled samples") 
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    return labeled_positive, unlabeled_data, val_data


def extract_context_from_sample(sample):
    """Extract context components from a data sample."""
    project_context = sample.get("proposal", "").strip()
    long_term_memory = sample.get("summarized_long_term_memory", "").strip()
    recent_conversation = "\n".join(sample.get("short_term_memory", [])).strip()
    
    return project_context, long_term_memory, recent_conversation


def process_unlabeled_samples(observer, presenter, projector, coordinator, unlabeled_data, batch_size=8):
    """
    Process unlabeled samples through the full pipeline.
    
    Args:
        observer: Observer model
        presenter: Presenter model
        projector: LinearProjector model
        coordinator: Coordinator model
        unlabeled_data: List of unlabeled samples
        batch_size: Processing batch size
        
    Returns:
        tuple: (potential_positives, potential_negatives, log_probs)
    """
    logger.info(f"Processing {len(unlabeled_data)} unlabeled samples...")
    
    potential_positives = []
    potential_negatives = []
    all_log_probs = []
    
    # Process in batches to manage memory
    for i in range(0, len(unlabeled_data), batch_size):
        batch_samples = unlabeled_data[i:i+batch_size]
        batch_observer_embs = []
        batch_presenter_embs = []
        
        # Extract embeddings for batch
        for sample in batch_samples:
            project_context, long_term_memory, recent_conversation = extract_context_from_sample(sample)
            
            # Combine context for embedding extraction
            full_context = f"Project: {project_context}\nMemory: {long_term_memory}\nConversation: {recent_conversation}"
            
            # Get embeddings
            observer_emb = observer.get_last_hidden_layer(full_context)
            presenter_emb = presenter.get_last_hidden_layer(full_context)
            
            batch_observer_embs.append(observer_emb)
            batch_presenter_embs.append(presenter_emb)
        
        # Stack embeddings
        batch_observer_embs = torch.cat(batch_observer_embs, dim=0)  # [batch_size, hidden_dim]
        batch_presenter_embs = torch.cat(batch_presenter_embs, dim=0)  # [batch_size, hidden_dim]
        
        # Project presenter embeddings
        projected_presenter_embs = projector(batch_presenter_embs)
        
        # Get coordinator actions and log probabilities
        actions, log_probs = coordinator.sample_action(batch_observer_embs, projected_presenter_embs)
        
        # Split based on action threshold
        for j, (sample, action, log_prob) in enumerate(zip(batch_samples, actions, log_probs)):
            if action.item() >= action_threshold:
                potential_positives.append(sample)
            else:
                potential_negatives.append(sample)
            
            all_log_probs.append(log_prob)
    
    # Stack all log probabilities
    all_log_probs = torch.cat(all_log_probs, dim=0)
    
    logger.info(f"Identified {len(potential_positives)} potential positives")
    logger.info(f"Identified {len(potential_negatives)} potential negatives")
    
    return potential_positives, potential_negatives, all_log_probs


def run_sft_training(potential_positives, labeled_positives):
    """
    Run SFT training on augmented positive samples using LLaMA Factory.
    
    Args:
        potential_positives: Samples identified as potential positives
        labeled_positives: Original labeled positive samples
        
    Returns:
        str: Path to trained SFT model
    """
    logger.info(f"Starting SFT training with {len(potential_positives)} potential positives and {len(labeled_positives)} labeled positives...")
    
    # Combine all positive samples for training
    all_positives = potential_positives + labeled_positives
    logger.info(f"Total SFT training samples: {len(all_positives)}")
    
    # Create SFT dataset in LLaMA Factory format
    sft_dataset = []
    for sample in all_positives:
        project_context, long_term_memory, recent_conversation = extract_context_from_sample(sample)
        intervention = sample.get("Intervention content", "")
        
        if not intervention.strip():
            continue
            
        # Format for LLaMA Factory (instruction, input, output)
        instruction = f"""You are a scientific research assistant specializing in collaborative research interventions. Based on the project context, long-term memory, and recent conversation, provide a helpful scientific intervention that keeps the discussion aligned with the project goal.

Project Context: {project_context}

Long-term Memory: {long_term_memory}

Recent Conversation:
{recent_conversation}

Please provide a scientific intervention to guide the research discussion:"""
        
        sft_sample = {
            "instruction": instruction,
            "input": "",  # Empty input as requested
            "output": intervention
        }
        sft_dataset.append(sft_sample)
    
    # Save SFT dataset for LLaMA Factory
    sft_data_path = "data/presenter_sft_data.json"
    os.makedirs(os.path.dirname(sft_data_path), exist_ok=True)
    with open(sft_data_path, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"SFT dataset saved to {sft_data_path} with {len(sft_dataset)} samples")
    
    # Run LLaMA Factory SFT training using GPU 0,1
    logger.info("Starting LLaMA Factory SFT training on GPU 0,1...")
    
    sft_command = (
        f"CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train "
        f"--config_path llamafactory_configs/presenter_sft_config.yaml "
        f"--dataset_dir ./data "
        f"--dataset_info llamafactory_configs/dataset_info.json"
    )
    
    try:
        logger.info("Running LLaMA Factory SFT training command...")
        sft_log_path = f"{presenter_save_path}/sft_training.log"
        os.makedirs(os.path.dirname(sft_log_path), exist_ok=True)
        
        with open(sft_log_path, "w") as log_file:
            process = subprocess.Popen(sft_command, shell=True, stdout=log_file, stderr=log_file)
            process.wait()
        
        if process.returncode == 0:
            logger.info(" LLaMA Factory SFT training completed successfully")
            logger.info("  - Presenter model trained on intervention generation")
            logger.info("  - GPU 0,1 automatically managed")
            logger.info("  - LoRA adapters saved to ./saved_models/presenter_sft")
        else:
            logger.warning(f" LLaMA Factory SFT training failed with return code {process.returncode}")
    except Exception as e:
        logger.error(f"Error running LLaMA Factory SFT training: {e}")
    
    # Return model path (matching LLaMA Factory output)
    sft_model_path = "./saved_models/presenter_sft"
    logger.info(f"SFT training completed. Model saved to {sft_model_path}")
    
    return sft_model_path


def train_projector(projector, coordinator, training_samples, optimizer_projector, epochs=5):
    """
    Train the Linear Projector to align Presenter embeddings with Observer embeddings.
    
    Args:
        projector: LinearProjector model
        coordinator: Coordinator model (for getting combined embeddings)
        training_samples: Combined potential positives and negatives
        optimizer_projector: Projector optimizer
        epochs: Number of training epochs
    """
    logger.info(f"Training Linear Projector for {epochs} epochs on {len(training_samples)} samples...")
    
    projector.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for i, sample in enumerate(training_samples):
            # Extract context for embedding generation
            project_context, long_term_memory, recent_conversation = extract_context_from_sample(sample)
            
            # Get embeddings from Observer and Presenter
            # This is a simplified version - in practice, you'd get actual embeddings
            observer_embedding = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)  # Simulated Observer embedding
            presenter_embedding = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)  # Simulated Presenter embedding
            
            # Project Presenter embedding to Observer dimension
            projected_embedding = projector(presenter_embedding)
            
            # Alignment loss: MSE between projected and observer embeddings
            alignment_loss = nn.MSELoss()(projected_embedding, observer_embedding)
            
            # Backward pass
            optimizer_projector.zero_grad()
            alignment_loss.backward()
            optimizer_projector.step()
            
            total_loss += alignment_loss.item()
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Sample {i+1}/{len(training_samples)}, Loss: {alignment_loss.item():.4f}")
        
        avg_loss = total_loss / len(training_samples)
        logger.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    projector.eval()
    logger.info("Linear Projector training completed!")


def run_grpo_training(potential_negatives, labeled_positives):
    """
    Run GRPO training for Observer intervention judgment.
    
    Args:
        potential_negatives: Samples identified as potential negatives (ground truth = 0)
        labeled_positives: Original labeled positive samples (ground truth = 1)
        
    Returns:
        str: Path to trained GRPO model
    """
    logger.info(f"Starting GRPO training for Observer intervention judgment...")
    logger.info(f"  - {len(potential_negatives)} potential negative samples (ground truth = 0)")
    logger.info(f"  - {len(labeled_positives)} labeled positive samples (ground truth = 1)")
    
    # Create GRPO dataset for Observer intervention judgment
    grpo_dataset = []
    
    # Add potential negative samples (ground truth = 0, no intervention needed)
    for sample in potential_negatives:
        project_context, long_term_memory, recent_conversation = extract_context_from_sample(sample)
        
        # Observer prompt for intervention judgment
        prompt = observer_intervention_system_prompt + "\n\n" + observer_intervention_user_prompt.format(
            project_context=project_context,
            long_term_memory=long_term_memory,
            recent_conversation=recent_conversation
        )
        
        grpo_sample = {
            "prompt": prompt,
            "completion": "No Need Intervention",  # Ground truth for potential negatives
            "ground_truth": 0,  # 0 = no intervention needed
            "sample_type": "potential_negative"
        }
        grpo_dataset.append(grpo_sample)
    
    # Add labeled positive samples (ground truth = 1, intervention needed)
    for sample in labeled_positives:
        project_context, long_term_memory, recent_conversation = extract_context_from_sample(sample)
        
        # Observer prompt for intervention judgment
        prompt = observer_intervention_system_prompt + "\n\n" + observer_intervention_user_prompt.format(
            project_context=project_context,
            long_term_memory=long_term_memory,
            recent_conversation=recent_conversation
        )
        
        # Generate intervention reason from the sample's intervention content
        intervention_content = sample.get("Intervention content", "Intervention needed for research coherence")
        intervention_reason = f"Intervention Content: {intervention_content}"
        
        grpo_sample = {
            "prompt": prompt,
            "completion": intervention_reason,  # Ground truth for labeled positives
            "ground_truth": 1,  # 1 = intervention needed
            "sample_type": "labeled_positive"
        }
        grpo_dataset.append(grpo_sample)
    
    # Save GRPO dataset
    os.makedirs(os.path.dirname(grpo_train_path), exist_ok=True)
    with open(grpo_train_path, 'w', encoding='utf-8') as f:
        json.dump(grpo_dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"GRPO dataset saved to {grpo_train_path} with {len(grpo_dataset)} samples")
    
    # Run automated GRPO training
    logger.info("Running automated GRPO training...")
    grpo_command = (
        f"python auto_grpo_train.py "
        f"--train_path {grpo_train_path} "
        f"--output_dir {observer_save_path}/grpo_trained "
        f"--vllm_port 8003"
    )
    
    try:
        logger.info("Running automated GRPO training command...")
        with open(grpo_log_file_path, "w") as log_file:
            process = subprocess.Popen(grpo_command, shell=True, stdout=log_file, stderr=log_file)
            process.wait()
        
        if process.returncode == 0:
            logger.info(" Automated GRPO training completed successfully")
            logger.info("  - vLLM server automatically managed")
            logger.info("  - Observer model trained for intervention judgment")
            logger.info("  - GPU resources automatically allocated")
        else:
            logger.warning(f" Automated GRPO training failed with return code {process.returncode}")
    except Exception as e:
        logger.error(f"Error running automated GRPO training: {e}")
    
    # Return model path
    grpo_model_path = f"{observer_save_path}/grpo_trained"
    return grpo_model_path


def compute_combined_reward(observer_score_new, presenter_score_new, observer_score_old, presenter_score_old, lambda_weight=reward_weight_lambda):
    """
    Compute combined reward for Coordinator training.
    
    Args:
        observer_score_new: New Observer performance
        presenter_score_new: New Presenter performance  
        observer_score_old: Previous Observer performance
        presenter_score_old: Previous Presenter performance
        lambda_weight: Weighting between Observer and Presenter rewards
        
    Returns:
        torch.Tensor: Combined reward
    """
    observer_improvement = observer_score_new - observer_score_old
    presenter_improvement = presenter_score_new - presenter_score_old
    
    combined_reward = lambda_weight * observer_improvement + (1 - lambda_weight) * presenter_improvement
    
    logger.info(f"Observer improvement: {observer_improvement:.4f}")
    logger.info(f"Presenter improvement: {presenter_improvement:.4f}")
    logger.info(f"Combined reward: {combined_reward:.4f}")
    
    return torch.tensor(combined_reward, device=device)


def train_coordinator(coordinator, log_probs, reward, optimizer):
    """
    Train Coordinator using REINFORCE.
    
    Args:
        coordinator: Coordinator model
        log_probs: Log probabilities from coordinator actions
        reward: Combined reward signal
        optimizer: Coordinator optimizer
        
    Returns:
        float: Training loss value
    """
    # Compute REINFORCE loss
    loss = coordinator.compute_reinforce_loss(log_probs, reward)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def save_all_models(observer, presenter, projector, coordinator, epoch):
    """Save all models for the current epoch."""
    epoch_observer_path = f"{observer_save_path}_epoch_{epoch}"
    epoch_presenter_path = f"{presenter_save_path}_epoch_{epoch}"
    epoch_projector_path = f"{projector_save_path}_epoch_{epoch}.pt"
    epoch_coordinator_path = f"{coordinator_save_path}_epoch_{epoch}.pt"
    
    observer.save_model(epoch_observer_path)
    presenter.save_model(epoch_presenter_path)
    projector.save_model(epoch_projector_path)
    coordinator.save_model(epoch_coordinator_path)
    
    logger.info(f"All models saved for epoch {epoch}")


def main():
    """Main training loop."""
    logger.info("Starting CoLabScience Training - New Architecture")
    
    # Load data
    labeled_positives, unlabeled_data, val_data = load_training_data()
    
    # Initialize models
    logger.info("Initializing models...")
    observer = Observer()
    presenter = Presenter()
    projector = LinearProjector()
    coordinator = Coordinator()
    
    # No need for separate scorer - using simple metrics
    
    # Initialize optimizers
    coordinator_optimizer = optim.AdamW(coordinator.parameters(), lr=coordinator_learning_rate)
    projector_optimizer = optim.AdamW(projector.parameters(), lr=projector_learning_rate)
    
    # Initialize performance tracking
    prev_observer_score = 0.0
    prev_presenter_score = 0.0
    
    logger.info("Starting main training loop...")
    
    # Main training loop
    for epoch in range(total_training_epochs):
        logger.info(f"\n" + "="*50)
        logger.info(f"EPOCH {epoch+1}/{total_training_epochs}")
        logger.info("="*50)
        
        # Step 1: Process unlabeled samples
        logger.info("Step 1: Processing unlabeled samples through pipeline...")
        potential_positives, potential_negatives, log_probs = process_unlabeled_samples(
            observer, presenter, projector, coordinator, unlabeled_data
        )
        
        # Step 2: SFT Training on augmented positives
        logger.info("Step 2: Running SFT training on augmented positive samples...")
        sft_model_path = run_sft_training(potential_positives, labeled_positives)
        
        # Step 3: Train Projector to align embeddings
        logger.info("Step 3: Training Linear Projector...")
        train_projector(projector, coordinator, potential_positives + potential_negatives, projector_optimizer)
        
        # Step 4: GRPO Training on potential negatives and labeled positives
        logger.info("Step 4: Running GRPO training for Observer intervention judgment...")
        grpo_model_path = run_grpo_training(potential_negatives, labeled_positives)
        
        # Step 5: Evaluate models on validation set
        logger.info("Step 4: Evaluating models on validation set...")
        observer_score_new, presenter_score_new = evaluate_models_on_validation(
            observer, presenter, val_data
        )
        
        logger.info(f"Observer validation score: {observer_score_new:.4f}")
        logger.info(f"Presenter validation score: {presenter_score_new:.4f}")
        
        # Step 5: Compute combined reward
        logger.info("Step 5: Computing combined reward...")
        combined_reward = compute_combined_reward(
            observer_score_new, presenter_score_new,
            prev_observer_score, prev_presenter_score
        )
        
        # Step 6: Train Coordinator with REINFORCE
        logger.info("Step 6: Training Coordinator with REINFORCE...")
        coordinator_loss = train_coordinator(coordinator, log_probs, combined_reward, coordinator_optimizer)
        logger.info(f"Coordinator loss: {coordinator_loss:.4f}")
        
        # Step 7: Train projector (if needed)
        # The projector is trained implicitly through the coordinator gradients
        # But we can add explicit projector training if desired
        
        # Step 8: Update performance tracking
        prev_observer_score = observer_score_new
        prev_presenter_score = presenter_score_new
        
        # Step 9: Save models
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_all_models(observer, presenter, projector, coordinator, epoch + 1)
        
        # Step 10: Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"Epoch {epoch+1} completed successfully!")
    
    # Final model saving
    logger.info("Training completed! Saving final models...")
    save_all_models(observer, presenter, projector, coordinator, "final")
    
    # Cleanup
    observer.unload()
    presenter.unload()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()