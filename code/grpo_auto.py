
"""
Automated GRPO Training Script
==============================

This script automatically manages the GRPO training process:
1. Auto-detects IP address
2. Starts vLLM server on GPU 3 with Observer model
3. Launches GRPO training on GPUs 4,5,6,7
4. Manages the complete training pipeline

Usage:
    python auto_grpo_train.py --train_path ../data/grpo_pn_data.json --output_dir GRPO_Observer
"""

import argparse
import os
import socket
import subprocess
import time
import signal
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_grpo_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoGRPOTrainer:
    """Automated GRPO Training Manager"""
    
    def __init__(self, train_path, output_dir, vllm_port=8003):
        self.train_path = train_path
        self.output_dir = output_dir
        self.vllm_port = vllm_port
        self.vllm_process = None
        self.training_process = None
        self.host_ip = self.get_host_ip()
        
        logger.info(f"AutoGRPOTrainer initialized:")
        logger.info(f"  Host IP: {self.host_ip}")
        logger.info(f"  Training data: {train_path}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  vLLM port: {vllm_port}")
    
    def get_host_ip(self):
        """Auto-detect the host IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            logger.warning(f"Could not auto-detect IP: {e}")
            # Fallback to localhost
            return "127.0.0.1"
    
    def check_gpu_availability(self):
        """Check if required GPUs are available"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            gpu_info = result.stdout.strip().split('\n')
            
            available_gpus = []
            for i, line in enumerate(gpu_info):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_idx = int(parts[0])
                    gpu_name = parts[1]
                    free_memory = int(parts[2])
                    available_gpus.append((gpu_idx, gpu_name, free_memory))
            
            logger.info(f"Available GPUs: {len(available_gpus)}")
            for gpu_idx, gpu_name, free_mem in available_gpus:
                logger.info(f"  GPU {gpu_idx}: {gpu_name} ({free_mem} MB free)")
            
            # Check if we have GPUs 3,4,5,6,7
            required_gpus = [3, 4, 5, 6, 7]
            available_indices = [gpu[0] for gpu in available_gpus]
            
            missing_gpus = [gpu for gpu in required_gpus if gpu not in available_indices]
            if missing_gpus:
                logger.warning(f"Missing required GPUs: {missing_gpus}")
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check GPU availability: {e}")
            return False
    
    def start_vllm_server(self):
        """Start vLLM server on GPU 3 with Observer model"""
        from config import observer_model_name
        
        logger.info("Starting vLLM server...")
        logger.info(f"  Model: {observer_model_name}")
        logger.info(f"  GPU: 3")
        logger.info(f"  Port: {self.vllm_port}")
        
        vllm_command = [
            "bash", "-c",
            f"CUDA_VISIBLE_DEVICES=3 NCCL_CUMEM_ENABLE=0 trl vllm-serve "
            f"--model {observer_model_name} "
            f"--tensor_parallel_size 1 "
            f"--port {self.vllm_port} "
            f"--gpu_memory_utilization 0.9 "
            f"--dtype bfloat16"
        ]
        
        try:
            # Start vLLM server in background
            self.vllm_process = subprocess.Popen(
                vllm_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"vLLM server started with PID: {self.vllm_process.pid}")
            
            # Wait for server to be ready
            self.wait_for_vllm_ready()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def wait_for_vllm_ready(self, timeout=300):
        """Wait for vLLM server to be ready"""
        import requests
        
        logger.info("Waiting for vLLM server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the health endpoint
                response = requests.get(f"http://{self.host_ip}:{self.vllm_port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(10)
            logger.info(f"Still waiting for vLLM server... ({int(time.time() - start_time)}s)")
        
        logger.error("vLLM server failed to start within timeout")
        return False
    
    def start_grpo_training(self):
        """Start GRPO training on GPUs 4,5,6,7"""
        logger.info("Starting GRPO training...")
        logger.info(f"  GPUs: 4,5,6,7")
        logger.info(f"  Training data: {self.train_path}")
        logger.info(f"  Output directory: {self.output_dir}")
        
        # Prepare training command
        training_command = [
            "bash", "-c",
            f"CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch "
            f"--config_file ./accelerate_configs/deepspeed_zero3.yaml "
            f"--num_processes 4 --num_machines 1 "
            f"--main_process_ip {self.host_ip} --machine_rank 0 "
            f"--rdzv_backend c10d grpo_trainer.py "
            f"--vllm_server_host {self.host_ip} "
            f"--train_path {self.train_path} "
            f"--output_dir {self.output_dir}"
        ]
        
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Start training
            log_file = f"{self.output_dir}/training.log"
            with open(log_file, "w") as f:
                self.training_process = subprocess.Popen(
                    training_command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            logger.info(f"GRPO training started with PID: {self.training_process.pid}")
            logger.info(f"Training logs: {log_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start GRPO training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor the training process"""
        logger.info("Monitoring training process...")
        
        try:
            # Wait for training to complete
            return_code = self.training_process.wait()
            
            if return_code == 0:
                logger.info("GRPO training completed successfully!")
            else:
                logger.error(f"GRPO training failed with return code: {return_code}")
            
            return return_code == 0
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes"""
        logger.info("Cleaning up processes...")
        
        # Stop training process
        if self.training_process and self.training_process.poll() is None:
            logger.info("Stopping training process...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
        
        # Stop vLLM server
        if self.vllm_process and self.vllm_process.poll() is None:
            logger.info("Stopping vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
        
        logger.info("Cleanup completed")
    
    def run(self):
        """Run the complete automated GRPO training pipeline"""
        try:
            logger.info("Starting Automated GRPO Training Pipeline")
            
            # Step 1: Check GPU availability
            if not self.check_gpu_availability():
                logger.error("Required GPUs not available")
                return False
            
            # Step 2: Start vLLM server
            if not self.start_vllm_server():
                logger.error("Failed to start vLLM server")
                return False
            
            # Step 3: Start GRPO training
            if not self.start_grpo_training():
                logger.error("Failed to start GRPO training")
                return False
            
            # Step 4: Monitor training
            success = self.monitor_training()
            
            if success:
                logger.info("Automated GRPO training completed successfully!")
            else:
                logger.error("Automated GRPO training failed")
            
            return success
            
        except KeyboardInterrupt:
            logger.info("Training pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in training pipeline: {e}")
            return False
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Automated GRPO Training")
    parser.add_argument("--train_path", type=str, required=True, 
                       help="Path to GRPO training data JSON file")
    parser.add_argument("--output_dir", type=str, default="GRPO_Observer",
                       help="Output directory for trained models")
    parser.add_argument("--vllm_port", type=int, default=8003,
                       help="Port for vLLM server")
    
    args = parser.parse_args()
    
    # Validate training data exists
    if not os.path.exists(args.train_path):
        logger.error(f"Training data file not found: {args.train_path}")
        return False
    
    # Create trainer and run
    trainer = AutoGRPOTrainer(
        train_path=args.train_path,
        output_dir=args.output_dir,
        vllm_port=args.vllm_port
    )
    
    # Setup signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        trainer.cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the training pipeline
    success = trainer.run()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)