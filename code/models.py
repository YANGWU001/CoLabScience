"""
CoLabScience Models - New Architecture
=====================================================
Architecture Components:
1. Observer (LLM): Observes and understands research discussion state
2. Presenter (LLM): Generates scientific interventions 
3. LinearProjector: Projects Presenter embeddings to Observer dimension
4. Coordinator (6-layer MLP): Makes intervention decisions based on combined embeddings

Training Flow:
- Sample → Observer/Presenter embeddings → Projector → Concat → Coordinator → Action probability
- Action >= 0.5: Potential positive (for SFT augmentation)
- Action < 0.5: Potential negative (for GRPO PN training)
- End-to-end training with combined rewards from Observer and Presenter performance
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

import json
import os

# Import configurations
from config import (
    observer_model_name, observer_hidden_dim, observer_learning_rate, observer_batch_size,
    presenter_model_name, presenter_hidden_dim, presenter_learning_rate, presenter_batch_size,
    projector_input_dim, projector_output_dim, projector_learning_rate,
    coordinator_input_dim, coordinator_hidden_dims, coordinator_output_dim, coordinator_dropout, coordinator_learning_rate,
    action_threshold, device, system_prompt_template, user_prompt_template,
    observer_task_instruction, presenter_task_instruction
)


class Observer(nn.Module):
    """
    Observer LLM for understanding research discussion state.
    Provides embeddings for coordinator decision making.
    """
    
    def __init__(self, model_name=observer_model_name, lora_path=None):
        super().__init__()
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model (avoiding all distributed issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=False,
            use_auth_token=False,
            revision="main"
        )
        # Move to device manually
        self.model.to(device)
        
        # Apply LoRA if path provided
        if lora_path and os.path.exists(lora_path):
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"[Observer] Loaded LoRA adapter from {lora_path}")
        else:
            # Prepare model for LoRA training
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.eval()
    
    def get_last_hidden_layer(self, text):
        """
        Extract last hidden layer embedding for coordinator input.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            torch.Tensor: Last hidden state [1, hidden_dim]
        """
        # Format prompt for observer task
        prompt = user_prompt_template.format(
            project_context="",
            long_term_memory="", 
            recent_conversation=text,
            task_instruction=observer_task_instruction
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state of the last token
            last_hidden_state = outputs.hidden_states[-1]
            # Use mean pooling over sequence length
            embedding = last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            
        return embedding
    
    def observe_and_analyze(self, project_context, long_term_memory, recent_conversation):
        """
        Generate observational analysis of the research discussion.
        
        Args:
            project_context (str): Research project context
            long_term_memory (str): Summarized long-term memory
            recent_conversation (str): Recent conversation to analyze
            
        Returns:
            str: Observational analysis
        """
        prompt = system_prompt_template + user_prompt_template.format(
            project_context=project_context,
            long_term_memory=long_term_memory,
            recent_conversation=recent_conversation,
            task_instruction=observer_task_instruction
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def save_model(self, save_path):
        """Save the Observer model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[Observer] Model saved to {save_path}")
    
    def unload(self):
        """Unload model to free memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()


class Presenter(nn.Module):
    """
    Presenter LLM for generating scientific interventions.
    Provides embeddings for coordinator and generates final interventions.
    """
    
    def __init__(self, model_name=presenter_model_name, lora_path=None):
        super().__init__()
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model (avoiding all distributed issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=False,
            use_auth_token=False,
            revision="main"
        )
        # Move to device manually
        self.model.to(device)
        
        # Apply LoRA if path provided
        if lora_path and os.path.exists(lora_path):
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"[Presenter] Loaded LoRA adapter from {lora_path}")
        else:
            # Prepare model for LoRA training
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.eval()
    
    def get_last_hidden_layer(self, text):
        """
        Extract last hidden layer embedding for coordinator input.
        
        Args:
            text (str): Input text to process
            
        Returns:
            torch.Tensor: Last hidden state [1, hidden_dim]
        """
        # Format prompt for presenter task
        prompt = user_prompt_template.format(
            project_context="",
            long_term_memory="",
            recent_conversation=text,
            task_instruction=presenter_task_instruction
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state of the last token
            last_hidden_state = outputs.hidden_states[-1]
            # Use mean pooling over sequence length
            embedding = last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            
        return embedding
    
    def generate_intervention(self, project_context, long_term_memory, recent_conversation):
        """
        Generate scientific intervention based on context.
        
        Args:
            project_context (str): Research project context
            long_term_memory (str): Summarized long-term memory
            recent_conversation (str): Recent conversation context
            
        Returns:
            str: Generated scientific intervention
        """
        prompt = system_prompt_template + user_prompt_template.format(
            project_context=project_context,
            long_term_memory=long_term_memory,
            recent_conversation=recent_conversation,
            task_instruction=presenter_task_instruction
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def save_model(self, save_path):
        """Save the Presenter model."""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[Presenter] Model saved to {save_path}")
    
    def unload(self):
        """Unload model to free memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()


class LinearProjector(nn.Module):
    """
    Linear projector to align Presenter embeddings with Observer embeddings.
    Projects Presenter hidden dimension to Observer hidden dimension.
    """
    
    def __init__(self, input_dim=projector_input_dim, output_dim=projector_output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Move to device and ensure correct dtype (matching LLM models)
        self.to(device)
        self.projection = self.projection.to(torch.bfloat16)
    
    def forward(self, presenter_embedding):
        """
        Project Presenter embedding to Observer dimension.
        
        Args:
            presenter_embedding (torch.Tensor): [batch_size, input_dim]
            
        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        return self.projection(presenter_embedding)
    
    def save_model(self, save_path):
        """Save the projector model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[LinearProjector] Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load the projector model."""
        if os.path.exists(load_path):
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"[LinearProjector] Model loaded from {load_path}")
        else:
            print(f"[LinearProjector] No saved model found at {load_path}")


class Coordinator(nn.Module):
    """
    6-layer MLP Coordinator for making intervention decisions.
    Takes concatenated Observer and projected Presenter embeddings as input.
    Outputs action probability for positive/negative classification.
    """
    
    def __init__(self, 
                 input_dim=coordinator_input_dim,
                 hidden_dims=coordinator_hidden_dims,
                 output_dim=coordinator_output_dim,
                 dropout=coordinator_dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build 6-layer MLP
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Sigmoid()  # Output probability
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Move to device and ensure correct dtype (matching LLM models)
        self.to(device)
        self.network = self.network.to(torch.bfloat16)
    
    def forward(self, observer_embedding, projected_presenter_embedding):
        """
        Forward pass through coordinator.
        
        Args:
            observer_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            projected_presenter_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            
        Returns:
            torch.Tensor: Action probability [batch_size, 1]
        """
        # Concatenate embeddings
        combined_embedding = torch.cat([observer_embedding, projected_presenter_embedding], dim=-1)
        
        # Pass through MLP
        action_prob = self.network(combined_embedding)
        
        return action_prob
    
    def sample_action(self, observer_embedding, projected_presenter_embedding):
        """
        Sample action and compute log probability for REINFORCE.
        
        Args:
            observer_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            projected_presenter_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            
        Returns:
            tuple: (action, log_prob)
                - action (torch.Tensor): Sampled action [batch_size, 1]
                - log_prob (torch.Tensor): Log probability [batch_size, 1]
        """
        prob = self.forward(observer_embedding, projected_presenter_embedding)
        
        # Sample action using Bernoulli
        action = torch.bernoulli(prob)
        
        # Compute log probability
        log_prob = action * torch.log(prob + 1e-8) + (1 - action) * torch.log(1 - prob + 1e-8)
        
        return action.int(), log_prob
    
    def predict_action(self, observer_embedding, projected_presenter_embedding, threshold=action_threshold):
        """
        Predict action based on threshold.
        
        Args:
            observer_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            projected_presenter_embedding (torch.Tensor): [batch_size, observer_hidden_dim]
            threshold (float): Threshold for positive/negative classification
            
        Returns:
            torch.Tensor: Predicted actions [batch_size, 1]
        """
        with torch.no_grad():
            prob = self.forward(observer_embedding, projected_presenter_embedding)
            actions = (prob >= threshold).int()
        
        return actions
    
    def compute_reinforce_loss(self, log_probs, reward):
        """
        Compute REINFORCE loss for policy gradient training.
        
        Args:
            log_probs (torch.Tensor): Log probabilities [batch_size, 1]
            reward (torch.Tensor or float): Reward signal
            
        Returns:
            torch.Tensor: Policy loss
        """
        if isinstance(reward, (float, int)):
            reward = torch.tensor(reward, device=device)
        
        # REINFORCE loss: -mean(log_prob * reward)
        loss = -torch.mean(log_probs * reward)
        
        return loss
    
    def save_model(self, save_path):
        """Save the coordinator model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"[Coordinator] Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load the coordinator model."""
        if os.path.exists(load_path):
            self.load_state_dict(torch.load(load_path, map_location=device))
            print(f"[Coordinator] Model loaded from {load_path}")
        else:
            print(f"[Coordinator] No saved model found at {load_path}")


# Utility Functions

def evaluate_models_on_validation(observer, presenter, val_dataset):
    """
    Evaluate Observer and Presenter performance on validation set.
    Uses simple accuracy-based metrics instead of BERTScore.
    
    Args:
        observer (Observer): Observer model
        presenter (Presenter): Presenter model  
        val_dataset (list): Validation dataset
        
    Returns:
        tuple: (observer_score, presenter_score)
    """
    observer_correct = 0
    presenter_correct = 0
    total_samples = 0
    
    for sample in val_dataset:
        # Extract context
        project_context = sample.get("proposal", "")
        long_term_memory = sample.get("summarized_long_term_memory", "")
        recent_conversation = "\n".join(sample.get("short_term_memory", []))
        ground_truth = sample.get("Intervention content", "")
        
        # Skip samples without ground truth
        if not ground_truth or ground_truth.strip() == "":
            continue
            
        total_samples += 1
        
        # Observer analysis
        try:
            observer_analysis = observer.observe_and_analyze(
                project_context, long_term_memory, recent_conversation
            )
            # Simple length-based evaluation (can be improved)
            if len(observer_analysis.strip()) > 10:  # Basic quality check
                observer_correct += 1
        except:
            pass
        
        # Presenter intervention
        try:
            presenter_intervention = presenter.generate_intervention(
                project_context, long_term_memory, recent_conversation
            )
            # Simple length-based evaluation (can be improved)
            if len(presenter_intervention.strip()) > 10:  # Basic quality check
                presenter_correct += 1
        except:
            pass
    
    # Compute simple accuracy scores
    observer_score = observer_correct / max(total_samples, 1)
    presenter_score = presenter_correct / max(total_samples, 1)
    
    return observer_score, presenter_score


