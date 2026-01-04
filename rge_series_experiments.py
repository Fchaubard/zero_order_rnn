#!/usr/bin/env python3
"""
In series RGE experiments on LSTM and DNC training on a few tasks: overfit, copy, sort, reverse, add, penn treebank, etc. 
python rge_series_experiments.py --unit_test --max_iterations 10 --micro_batch_size 1 --seq_length 10 --device cuda:1 --solver 1SPSA

 choices=["BPTT", "1SPSA", "1.5-SPSA", "2SPSA" ... ] )
"""

import argparse
import os
import random
import string
import time
import math
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from flashrnn.flashrnn import flashrnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from simpletokenizers.simpletokenizers import CharTokenizer, NumericTokenizer, get_tiktoken
from models.models        import LSTM, DNC #Transformer, Mamba, SSM
from tasks.tasks          import get_examples_for_task, compute_task_loss, compute_task_accuracy

import pdb

# Optional imports
try:
    import tiktoken  # Used only if tokenizer='hf_tiktoken' is specified
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    # Requires:  flashrnn ≥ 0.1  (pip install flash-rnn or pip install -e . from repo RECOMMENDED!) OR it will fallback to pytroch and be slow.
    from flashrnn.flashrnn import flashrnn           # official package name
    FLASH_OK = True
except ModuleNotFoundError:
    FLASH_OK = False
    warnings.warn("flashrnn not found – LSTM will fall back to PyTorch LSTM.",
                  RuntimeWarning)

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. Experiment tracking will be disabled.")
    print("To enable, install with: pip install wandb")

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    """Simple logging helper"""
    print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Binary LR Search Functions
# ─────────────────────────────────────────────────────────────────────────────

def probe_loss_at_lr(model_params, loss_fn, lr, epsilon, num_perturbations,
                     distribution, args, seed=0, n_steps=1,
                     cache_weights_to_cpu=False):
    """
    Probe the loss after taking n_steps with a given learning rate.

    Saves current weights, takes n_steps at the given lr/epsilon,
    evaluates loss, then restores original weights.

    Args:
        model_params: List of model parameter tensors
        loss_fn: Callable that returns the current loss
        lr: Learning rate to probe (also used as epsilon)
        epsilon: Epsilon value (typically same as lr for SPSA)
        num_perturbations: Number of perturbations per step
        distribution: Perturbation distribution ('rad', 'normal', etc.)
        args: Full args namespace
        seed: Random seed for reproducibility
        n_steps: Number of optimization steps to take
        cache_weights_to_cpu: If True, save weights to CPU to save GPU memory

    Returns:
        Final loss after n_steps at the given lr
    """
    device = model_params[0].device

    # Save current weights
    if cache_weights_to_cpu:
        saved_weights = [p.data.clone().cpu() for p in model_params]
    else:
        saved_weights = [p.data.clone() for p in model_params]

    # Temporarily modify args
    original_lr = args.learning_rate
    original_eps = args.epsilon
    original_cache_gradients = args.cache_gradients if hasattr(args, 'cache_gradients') else True
    args.learning_rate = lr
    args.epsilon = lr  # For SPSA, epsilon typically equals lr
    # Force cache_gradients=True during probing (the non-cached branch is deprecated)
    args.cache_gradients = True

    # Take n_steps
    for step in range(n_steps):
        # Set seed for reproducibility across probes
        torch.manual_seed(seed + step)
        random.seed(seed + step)

        if args.solver == "1.5-SPSA":
            SPSA1_5(
                model_params=model_params,
                loss_fn=loss_fn,
                lr=lr,
                epsilon=lr,
                iterations=1,
                num_perturbations=num_perturbations,
                distribution=distribution,
                antithetic=args.antithetic_sampling,
                use_adaptive_step=False,
                clip_grad_norm=args.grad_clip,
                cache_gradients=True,  # Always use cached path (non-cached is deprecated)
                args=args,
            )
        elif args.solver == "1SPSA":
            cdrge_optimize(
                model_params=model_params,
                loss_fn=loss_fn,
                lr=lr,
                epsilon=lr,
                iterations=1,
                num_perturbations=num_perturbations,
                distribution=distribution,
                antithetic=args.antithetic_sampling,
                use_adaptive_step=False,
                clip_grad_norm=args.grad_clip,
                cache_gradients=True,  # Always use cached path (non-cached is deprecated)
                args=args,
            )
        else:
            raise ValueError(f"Binary LR search not supported for solver: {args.solver}")

    # Evaluate loss after steps
    with torch.no_grad():
        final_loss = loss_fn()
        if hasattr(final_loss, 'item'):
            final_loss = final_loss.item()

    # Restore original weights
    for p, saved in zip(model_params, saved_weights):
        if cache_weights_to_cpu:
            p.data.copy_(saved.to(device))
        else:
            p.data.copy_(saved)

    # Restore original args
    args.learning_rate = original_lr
    args.epsilon = original_eps
    args.cache_gradients = original_cache_gradients

    return final_loss


def binary_search_lr(model_params, loss_fn, lr_min, lr_max,
                     num_perturbations, distribution, args,
                     depth=3, seed=0, probe_steps=1,
                     cache_weights_to_cpu=False):
    """
    Binary search for optimal lr in log space.

    Depth 1: Test 3 points (min, mid, max)
    Depth 2: Halve to better half, test midpoint
    Depth 3: Halve again, test midpoint

    Args:
        model_params: List of model parameter tensors
        loss_fn: Callable that returns current loss
        lr_min: Minimum learning rate to search
        lr_max: Maximum learning rate to search
        num_perturbations: Number of perturbations per step
        distribution: Perturbation distribution
        args: Full args namespace
        depth: Search depth (default 3)
        seed: Random seed for probing
        probe_steps: Number of steps to take when probing each LR
        cache_weights_to_cpu: Save weights to CPU during probing

    Returns:
        best_lr: Best learning rate found
    """
    # Verify restoration works before expensive search
    initial_loss = loss_fn()
    if hasattr(initial_loss, 'item'):
        initial_loss = initial_loss.item()

    _ = probe_loss_at_lr(model_params, loss_fn, lr_min, lr_min,
                         num_perturbations, distribution, args,
                         seed=999, n_steps=1,
                         cache_weights_to_cpu=cache_weights_to_cpu)

    restored_loss = loss_fn()
    if hasattr(restored_loss, 'item'):
        restored_loss = restored_loss.item()

    if abs(restored_loss - initial_loss) > 0.01:
        log(f"  WARNING: Weight restoration drift: {initial_loss:.4f} -> {restored_loss:.4f}")

    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)

    # Depth 1: Test 3 points
    log_mid = (log_min + log_max) / 2
    points = [log_min, log_mid, log_max]
    results = []

    steps_str = f" ({probe_steps} steps each)" if probe_steps > 1 else ""
    log(f"  Binary search depth 1: testing {len(points)} points in [{lr_min:.0e}, {lr_max:.0e}]{steps_str}")

    for log_lr in points:
        lr = 10 ** log_lr
        loss = probe_loss_at_lr(model_params, loss_fn, lr, lr,
                                num_perturbations, distribution, args,
                                seed=seed, n_steps=probe_steps,
                                cache_weights_to_cpu=cache_weights_to_cpu)
        results.append((log_lr, loss))
        log(f"    lr={lr:.2e} -> loss={loss:.4f}")

    # Find best and narrow search
    best_idx = min(range(len(results)), key=lambda i: results[i][1])

    for d in range(2, depth + 1):
        # Narrow to region around best
        if best_idx == 0:
            log_max = results[1][0]
        elif best_idx == len(results) - 1:
            log_min = results[-2][0]
        else:
            log_min = results[best_idx - 1][0]
            log_max = results[best_idx + 1][0]

        # Test midpoint of new range
        log_mid = (log_min + log_max) / 2
        lr = 10 ** log_mid
        loss = probe_loss_at_lr(model_params, loss_fn, lr, lr,
                                num_perturbations, distribution, args,
                                seed=seed, n_steps=probe_steps,
                                cache_weights_to_cpu=cache_weights_to_cpu)
        log(f"  Binary search depth {d}: lr={lr:.2e} -> loss={loss:.4f}")

        # Update results
        results.append((log_mid, loss))
        results.sort(key=lambda x: x[0])
        best_idx = min(range(len(results)), key=lambda i: results[i][1])

    best_log_lr, best_loss = min(results, key=lambda x: x[1])
    best_lr = 10 ** best_log_lr
    log(f"  Binary search complete: best lr={best_lr:.2e} (loss={best_loss:.4f})")
    return best_lr


class EMAPlateauDetector:
    """
    Detects when training loss has plateaued using exponential moving average.

    Triggers when the EMA hasn't improved by more than threshold for
    patience iterations.
    """
    def __init__(self, alpha=0.1, patience=10, threshold=0.01):
        """
        Args:
            alpha: EMA smoothing factor (higher = more weight on recent values)
            patience: Number of iterations without improvement before triggering
            threshold: Minimum relative improvement required
        """
        self.alpha = alpha
        self.patience = patience
        self.threshold = threshold
        self.ema = None
        self.best_ema = None
        self.iters_without_improvement = 0

    def update(self, loss):
        """
        Update EMA with new loss value.

        Args:
            loss: Current training loss

        Returns:
            True if plateau detected, False otherwise
        """
        if self.ema is None:
            self.ema = loss
            self.best_ema = loss
            return False

        # Update EMA
        self.ema = self.alpha * loss + (1 - self.alpha) * self.ema

        # Check for improvement
        relative_improvement = (self.best_ema - self.ema) / (self.best_ema + 1e-8)

        if relative_improvement > self.threshold:
            self.best_ema = self.ema
            self.iters_without_improvement = 0
        else:
            self.iters_without_improvement += 1

        return self.iters_without_improvement >= self.patience

    def reset(self):
        """Reset the detector after LR search"""
        self.iters_without_improvement = 0
        # Keep EMA and best_ema to continue tracking


def compute_task_loss(logits, ids_np, tok, task, verbose=False):
      """Compute loss with gradients based on task type with proper shift for predictions"""
      B, T, V = logits.shape
      device = logits.device

      # Different separator tokens for different tasks
      if task in ["copy", "repeat_copy", "sort", "reverse"]:
          sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
      elif task == "add":
          sep = tok.token_to_id.get("=", 0)
      else:  # penn_tree_bank or any others
          # For PTB, predict next token for each position
          targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
          shifted_logits = logits[:, :-1, :]
          loss = torch.nn.functional.cross_entropy(
              shifted_logits.reshape(-1, V),
              targets.reshape(-1),
              reduction='mean'
          )
          return loss

      # Get space token ID
      space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)

      # Initialize running total and count
      total_loss = torch.tensor(0.0, device=device, requires_grad=True)
      count = 0

      for b in range(B):
          try:
              # Find separator position
              pos = list(ids_np[b]).index(sep)
          except ValueError:
              continue

          # Get output part (everything after the separator)
          output_part = ids_np[b][pos + 1:]

          if len(output_part) <= 1:  # Need at least 2 tokens for prediction
              continue

          # Find content indices in output (non-space tokens)
          content_indices = [i for i, t in enumerate(output_part) if t != space_id]
          if not content_indices:
              continue

          # Debug prints
          if verbose and b < 3:
              input_text = tok.decode(ids_np[b][:pos+1])
              target_text = tok.decode([output_part[i] for i in content_indices if i < len(output_part)])
              print(f"\nSample {b}:")
              print(f"  Input: '{input_text}'")
              print(f"  Target: '{target_text}'")

          # The crucial correction: 
          # For each position i, use logits at position pos+1+i to predict token at position i+1
          pred_logits = []
          tgt_tokens = []

          # For each position except the last one in output_part
          for i in range(len(output_part) - 1):
              # Only include if it's a content token or space following content
              if (i in content_indices) or (i-1 in content_indices and output_part[i] == space_id):
                  if pos + 1 + i < T:  # Ensure we're within logits sequence length
                      pred_logits.append(logits[b, pos + 1 + i])
                      tgt_tokens.append(output_part[i + 1])  # Predict NEXT token

          # Skip if we have no valid targets
          if not tgt_tokens:
              continue

          # Stack logits and prepare targets
          if pred_logits:
              stacked_logits = torch.stack(pred_logits)
              targets = torch.tensor(tgt_tokens, device=device, dtype=torch.long)

              # stacked_logits = stacked_logits.to(torch.float64)   # <— or .float() or nothing.. not sure it really helps.
              stacked_logits = stacked_logits.to(torch.float32) 
              # Compute batch loss
              batch_loss = torch.nn.functional.cross_entropy(
                  stacked_logits,
                  targets,
                  reduction='sum'
              )

              total_loss = total_loss + batch_loss
              count += len(tgt_tokens)

      # If no valid samples, return dummy loss
      if count == 0:
          return logits.sum() * 0.0

      return total_loss / count

def compute_task_loss_with_gradients(logits, ids_np, tok, task, verbose=False):
      """Compute loss with gradients based on task type with proper shift for predictions"""
      B, T, V = logits.shape
      device = logits.device

      # Different separator tokens for different tasks
      if task in ["copy", "repeat_copy", "sort", "reverse"]:
          sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
      elif task == "add":
          sep = tok.token_to_id.get("=", 0)
      else:  # penn_tree_bank or any others
          # For PTB, predict next token for each position
          targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
          shifted_logits = logits[:, :-1, :]
          loss = torch.nn.functional.cross_entropy(
              shifted_logits.reshape(-1, V),
              targets.reshape(-1),
              reduction='mean'
          )
          return loss

      # Get space token ID
      space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)

      # Initialize running total and count
      total_loss = torch.tensor(0.0, device=device, requires_grad=True)
      count = 0

      for b in range(B):
          try:
              # Find separator position
              pos = list(ids_np[b]).index(sep)
          except ValueError:
              continue

          # Get output part (everything after the separator)
          output_part = ids_np[b][pos + 1:]

          if len(output_part) <= 1:  # Need at least 2 tokens for prediction
              continue

          # Find content indices in output (non-space tokens)
          content_indices = [i for i, t in enumerate(output_part) if t != space_id]
          if not content_indices:
              continue

          # Debug prints
          if verbose and b < 3:
              input_text = tok.decode(ids_np[b][:pos+1])
              target_text = tok.decode([output_part[i] for i in content_indices if i < len(output_part)])
              print(f"\nSample {b}:")
              print(f"  Input: '{input_text}'")
              print(f"  Target: '{target_text}'")

          # The crucial correction: 
          # For each position i, use logits at position pos+1+i to predict token at position i+1
          pred_logits = []
          tgt_tokens = []

          # For each position except the last one in output_part
          for i in range(len(output_part) - 1):
              # Only include if it's a content token or space following content
              if (i in content_indices) or (i-1 in content_indices and output_part[i] == space_id):
                  if pos + 1 + i < T:  # Ensure we're within logits sequence length
                      pred_logits.append(logits[b, pos + 1 + i])
                      tgt_tokens.append(output_part[i + 1])  # Predict NEXT token

          # Skip if we have no valid targets
          if not tgt_tokens:
              continue

          # Stack logits and prepare targets
          if pred_logits:
              stacked_logits = torch.stack(pred_logits)
              targets = torch.tensor(tgt_tokens, device=device, dtype=torch.long)

              # Compute batch loss
              batch_loss = torch.nn.functional.cross_entropy(
                  stacked_logits,
                  targets,
                  reduction='sum'
              )

              total_loss = total_loss + batch_loss
              count += len(tgt_tokens)

      # If no valid samples, return dummy loss
      if count == 0:
          return logits.sum() * 0.0

      return total_loss / count

def compute_task_accuracy(logits, ids_np, tok, task, verbose=False):
    """Compute accuracy based on task type"""
    B, T, V = logits.shape
    device = logits.device
    
    total_correct = 0
    total_tokens = 0
    content_correct = 0
    content_tokens = 0
    eos_correct = 0
    eos_tokens = 0
    
    # Different separator tokens for different tasks
    if task in ["copy", "repeat_copy", "sort", "reverse"]:
        sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
    elif task == "add":
        sep = tok.token_to_id.get("=", 0)
    else:  # penn_tree_bank or any others
        # For PTB, predict next token for each position
        targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
        predictions = logits[:, :-1, :].argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
    
    # Get space token ID
    space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
    
    for b in range(B):
        try:
            # Find separator position
            pos = list(ids_np[b]).index(sep)
        except ValueError:
            continue
        
        # Get target tokens after separator
        output_part = ids_np[b][pos+1:]
        
        # Find content tokens (non-space) and add EOS token
        content_indices = [i for i, t in enumerate(output_part) if t != space_id]
        
        if not content_indices:
            continue
        
        # Add one more index after the last content token to include an EOS space
        last_content_idx = max(content_indices)
        eos_idx = last_content_idx + 1
            
        # Get all predictions for the output part
        pred_logits = logits[b, pos+1:pos+1+len(output_part)]
        pred_ids = pred_logits.argmax(dim=-1).cpu().numpy()
        
        # Debug prints for verifying processing
        if verbose and b < 3:  # Limit debug to first 3 samples
            input_part = ids_np[b][:pos+1]
            input_text = tok.decode(input_part)
            
            # Decode actual target and prediction
            content_target = [output_part[i] for i in content_indices if i < len(output_part)]
            content_pred = [pred_ids[i] for i in content_indices if i < len(pred_ids)]
            
            target_text = tok.decode(content_target)
            pred_text = tok.decode(content_pred)
            
            # Include EOS token if it exists
            full_target = content_target.copy()
            full_pred = content_pred.copy()
            
            if eos_idx < len(output_part):
                full_target.append(output_part[eos_idx])
            if eos_idx < len(pred_ids):
                full_pred.append(pred_ids[eos_idx])
                
            full_target_text = tok.decode(full_target)
            full_pred_text = tok.decode(full_pred)
            
            print(f"\nAccuracy Sample {b}:")
            print(f"  Input: '{input_text}'")
            print(f"  Target (content): '{target_text}'")
            print(f"  Pred (content): '{pred_text}'")
            print(f"  Target (with EOS): '{full_target_text}'")
            print(f"  Pred (with EOS): '{full_pred_text}'")
        
        # Count correct content token predictions
        for i in content_indices:
            if i < len(pred_ids):  # Make sure we're in bounds
                total_tokens += 1
                content_tokens += 1
                if pred_ids[i] == output_part[i]:
                    total_correct += 1
                    content_correct += 1
                    
        # Count correct EOS token prediction if it exists
        if eos_idx < len(output_part) and eos_idx < len(pred_ids):
            total_tokens += 1
            eos_tokens += 1
            if pred_ids[eos_idx] == output_part[eos_idx]:
                total_correct += 1
                eos_correct += 1
    
    # Report detailed stats if verbose
    if verbose:
        content_acc = content_correct / max(content_tokens, 1) * 100
        eos_acc = eos_correct / max(eos_tokens, 1) * 100
        print(f"\nDetailed accuracy stats:")
        print(f"  Content tokens: {content_correct}/{content_tokens} = {content_acc:.2f}%")
        print(f"  EOS tokens: {eos_correct}/{eos_tokens} = {eos_acc:.2f}%")
        print(f"  Overall: {total_correct}/{total_tokens} = {total_correct / max(total_tokens, 1) * 100:.2f}%")
    
    return total_correct / max(total_tokens, 1)

# ───────── Memory Estimation ─────────
def estimate_memory_usage(model_params, B, T, G, N, D, V, S):
    """
    Estimate VRAM usage for the model

    Args:
        model_params: List of model parameters
        B: Batch size
        T: Sequence length
        G: Number of gates
        N: Number of heads
        D: Dimension per head
        V: Vocabulary size
        S: Number of states (2 for LSTM)

    Returns:
        param_memory: Memory used by parameters (in MB)
        activation_memory: Memory used by activations (in MB)
        total_memory: Total memory usage (in MB)
    """
    # Count parameter memory
    param_count = sum(p.numel() for p in model_params)
    param_memory = param_count * 2 / (1024 * 1024)  # bfloat16 = 2 bytes, convert to MB

    # Estimate activation memory
    # Key activations:
    # 1. One-hot encoding: B * T * V * 2 bytes
    # 2. Input embeddings: B * T * G * N * D * 2 bytes
    # 3. Hidden states: S * B * T * N * D * 2 bytes
    # 4. Logits: B * T * V * 2 bytes

    activation_bytes = (
        B * T * V * 2 +                # One-hot encoding
        B * T * G * N * D * 2 +        # Input embeddings
        S * B * T * N * D * 2 +        # Hidden states
        B * T * V * 2                  # Logits
    )

    activation_memory = activation_bytes / (1024 * 1024)  # Convert to MB
    total_memory = param_memory + activation_memory

    return param_memory, activation_memory, total_memory

# ───────── Parameter Perturbation and Optimization ─────────
def generate_rademacher_noise(param, epsilon):
    # 1) make an int8 tensor of 0 / 1
    noise = torch.randint(
        0, 2,                        # values {0,1}
        param.shape,
        device=param.device,
        dtype=torch.int8
    )
    # 2) map {0,1} → {-1,+1}  and cast to param.dtype (fp16)
    noise = noise.mul(2).sub_(1).to(param.dtype)

    # 3) scale by epsilon
    return noise.mul_(epsilon)
    
def generate_perturbation(param, epsilon, distribution='rad', seed=None):
    """Generate perturbation according to the specified distribution with optional seed"""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    if distribution == 'rad':
        # Rademacher distribution (±1)
        return generate_rademacher_noise(param, epsilon)
    elif distribution == 'normal':
        # Normal distribution
        return torch.randn_like(param) * epsilon
    elif distribution == 'uniform':
        # Uniform distribution in [-1, 1]
        return (torch.rand_like(param) * 2 - 1) * epsilon
    else:
        raise ValueError(f"Unknown distribution: {distribution}")



def SPSA1_5(model_params: List[torch.Tensor],
            loss_fn,
            lr: float,
            epsilon: float,
            iterations: int,
            num_perturbations: int = 20,
            distribution: str = 'rad',
            antithetic: bool = False,
            use_adaptive_step: bool = False,
            clip_grad_norm: float = 0.0,
            cache_gradients: bool = False,
            lambda_reg: float = 1., #1e-2,
            CHUNK_SIZE: int = 2**15,
            args=None):
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    
    gradients = [torch.zeros_like(p.data) for p in model_params]


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0

    if cache_gradients:
        # one tensor per param to accumulate Σ coeff·probe
        grad_buffer = [torch.zeros_like(p.data) for p in model_params]
    else:
        grad_estimate_list = []          # original storage
    
    
    if is_verbose:
        print("starting perturbations")

    total_clean_loss = 0.0
    for _ in range(macro_batch_size):
        clean_loss_batch = loss_fn()
        batch_loss = clean_loss_batch.item() if hasattr(clean_loss_batch, 'item') else float(clean_loss_batch)
        total_clean_loss += batch_loss

    clean_loss = total_clean_loss / macro_batch_size
    
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,                
                    epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            
            # Generate and apply perturbation
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        
        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,               
                     -2*epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            # Generate and apply perturbation
            # probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = -1*(pos_loss - neg_loss) / (2*num_perturbations) # no epsilon bc we mult later so it cancels out
        curv = abs(pos_loss -2*clean_loss + neg_loss)/(epsilon**2) 
        # curvature = max( 0.001 * curv, lambda_reg)
        curvature = max( curv**args.saturating_alpha, lambda_reg)
        
        # curvature = max( np.log(curv + 1e-8), lambda_reg)
        grad_estimate_list.append(grad_estimate / curvature)
        sum_of_losses += (pos_loss + neg_loss) / (2)

        if is_verbose:
          print(f"  grad_estimate {grad_estimate} curv {curv} denom {curvature} ")
        
        # for param_idx, param in enumerate(model_params):
        #     for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
        #         probe = generate_perturbation(
        #             chunk,               
        #             1,
        #             distribution,
        #             seed=pert_seed + param_idx
        #         )
        #         chunk.add_(probe,alpha=epsilon)         # in-place update of that slice only
        #         if cache_gradients:
        #             # coeff / epsilon  =  grad_estimate / (+ε)
        #             grad_chunk = grad_buffer[param_idx][chunk]      # same view
        #             grad_chunk.add_(probe, alpha=grad_estimate)

        for param_idx, param in enumerate(model_params):
            rows = param.size(0)
        
            for start in range(0, rows, CHUNK_SIZE):
                end   = min(start + CHUNK_SIZE, rows)
        
                slice_param = param.data[start:end]        # view onto weights
                probe = generate_perturbation(
                            slice_param,                   # same shape
                            1,
                            distribution,
                            seed=pert_seed + param_idx
                        )
        
                slice_param.add_(probe, alpha=epsilon)     # restore weights
        
                if cache_gradients:
                    # accumulate coeff·probe into the matching slice of grad_buffer
                    grad_buffer[param_idx][start:end].add_(probe,
                                                           alpha=grad_estimate)
            
            # # Regenerate the same noise
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)

            # # Restore parameter by adding the perturbation
            # param.data.add_(probe)
            

    # for idx, param in enumerate(model_params):
    #     # if idx == 3:                      # skip states tensor
    #     #     continue
    
    #     # -------- accumulate gradient in FP32 -------------------------
    #     grad = None
    #     for j in range(num_perturbations):
    #         seed         = all_seeds[j] + idx
    #         coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
    #         probe   = generate_perturbation(
    #                            param, coeff, distribution, seed)
            
    #         grad = probe if grad is None else grad + probe

    #     param.data.add_(grad, alpha= -1./float(num_perturbations)   )
    #     # -------- SGD step in FP32, then cast back -------------------
    #     # param_fp32 = param.data.float()
    #     # param_fp32.add_( -grad )                          # **NO extra ε/n**
    #     # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16
    # # SGD update (same as before)
    # param.data.add_(grad, alpha=-1. / float(num_perturbations))

    if cache_gradients:
        for buf, param in zip(grad_buffer, model_params):
            param.data.add_(buf) 

    else: # calculate this post from the seeds
        for idx, param in enumerate(model_params):
            # grad = None
            for j in range(num_perturbations):
                seed  = all_seeds[j] + idx
                coeff = grad_estimate_list[j]                 # (f⁺−f⁻)/2n # no ε bc ε == lr and they cancel out
        
                # ── NEW: accumulate probe chunk-by-chunk ─────────────────────
                for chunk in torch.split(param, CHUNK_SIZE, dim=0):     # 8k rows
                    probe = generate_perturbation(
                                chunk, coeff, distribution, seed)
                    chunk.add_(probe) 
        
                # grad = probe if grad is None else grad + probe


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = clean_loss
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics

    
def cdrge_optimize(model_params, 
                   loss_fn, 
                   lr, 
                   epsilon, 
                   iterations, 
                   num_perturbations=20,
                   distribution='rad', 
                   antithetic=False, 
                   use_adaptive_step=False, 
                   clip_grad_norm=0.0,
                   cache_gradients = True,
                   CHUNK_SIZE = 2**15, # TUNE THIS TO MAKE IT FIT IN VRAM BUT ALSO BE FAST
                   args=None):
    """Central Difference Random Gradient Estimation optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (recomend to be same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }


    lr_to_eta_ratio = lr/epsilon
    
    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0

    
    if args.beta1>0 and not hasattr(args, "coordinate_momentum"):                        # Nesterov buffer (m starts at 0)
        # coordinate_momentum   = [torch.zeros_like(p.data) for p in model_params]
        args.coordinate_momentum = [
                            list(torch.split(torch.zeros_like(p.data, dtype=torch.float32), CHUNK_SIZE, dim=0))
                            for p in model_params
                        ]
        
    if args.beta2>0 and not hasattr(args, "coordinate_variance"):                         # RMSProp accumulator (v starts at 1)
        # coordinate_variance = [torch.ones_like(p.data) for p in model_params]
        args.coordinate_variance = [
                                    list(torch.split(torch.ones_like(p.data, dtype=torch.float32), CHUNK_SIZE, dim=0))
                                    for p in model_params
                                ]


    if args.beta1>0 or args.beta2>0:
        cache_gradients = True  # theoretically, you dont need to have this, but not implemented otherwise..
                                # ensures useability for now.. 
                                # use unless you want to implement the non-cache version to save VRAM
    
    if cache_gradients:
        # one tensor per param to accumulate Σ coeff·probe
        # grad_buffer = [torch.zeros_like(p.data) for p in model_params]
        grad_buffer = [
                            list(torch.split(torch.zeros_like(p.data, dtype=torch.float32), CHUNK_SIZE, dim=0))
                            for p in model_params
                        ]

        
    else:
        grad_estimate_list = []          # original storage
    
    
    if is_verbose:
        print("starting perturbations")
        
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk_idx, chunk in enumerate(torch.split(param.data, CHUNK_SIZE, dim=0)):   # 8 k rows each
                
                probe = generate_perturbation(
                    chunk,                
                    1,
                    distribution,
                    seed=pert_seed + param_idx + chunk_idx
                    )
                if args.beta2>0 and args.use_probe_preconditioning:
                    # we need to adjust the probe by 1/coordinate_variance but in a safe way..
                    probe.div_(args.coordinate_variance[param_idx][chunk_idx].sqrt().add_(1e-8))
                    # probe /= torch.clamp_min(coordinate_variance[param_idx][chunk_idx], 1.0)
                probe *= epsilon
                chunk.add_(probe)         # in-place update of that slice only
            
            # Generate and apply perturbation
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        
        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk_idx, chunk in enumerate(torch.split(param.data, CHUNK_SIZE, dim=0)):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,                
                    -2,
                    distribution,
                    seed=pert_seed + param_idx + chunk_idx
                    )
                if args.beta2>0 and args.use_probe_preconditioning:
                    # we need to adjust the probe by 1/coordinate_variance but in a safe way..
                    probe.div_(args.coordinate_variance[param_idx][chunk_idx].sqrt().add_(1e-8))
                    # probe /= torch.clamp_min(coordinate_variance[param_idx][chunk_idx], 1.0)
                probe *= epsilon
                chunk.add_(probe)         # in-place update of that slice only

            # Generate and apply perturbation
            # probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = -1*(pos_loss - neg_loss) / (2*num_perturbations) # no epsilon bc we mult later so it cancels out
        sum_of_losses += (pos_loss + neg_loss) / (2)
        grad_estimate_list.append(grad_estimate)
        
        # for param_idx, param in enumerate(model_params):
        #     for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
        #         probe = generate_perturbation(
        #             chunk,               
        #             1,
        #             distribution,
        #             seed=pert_seed + param_idx
        #         )
        #         chunk.add_(probe,alpha=epsilon)         # in-place update of that slice only
        #         if cache_gradients:
        #             # coeff / epsilon  =  grad_estimate / (+ε)
        #             grad_chunk = grad_buffer[param_idx][chunk]      # same view
        #             grad_chunk.add_(probe, alpha=grad_estimate)

        # return theta to original values for next perturbation or for full updating
        for param_idx, param in enumerate(model_params):
            for chunk_idx, chunk in enumerate(torch.split(param.data, CHUNK_SIZE, dim=0)):
                
                probe = generate_perturbation(
                            chunk,                   # same shape
                            1,
                            distribution,
                            seed=pert_seed + param_idx + chunk_idx
                        )
                
                if args.beta2>0 and args.use_probe_preconditioning:
                    # we need to adjust the probe by 1/coordinate_variance but in a safe way..
                    probe.div_(args.coordinate_variance[param_idx][chunk_idx].sqrt().add_(1e-8))
                    # probe /= torch.clamp_min(coordinate_variance[param_idx][chunk_idx], 1.0)
                    
                     
                chunk.add_(probe, alpha=epsilon)     # restore weights
        
                if cache_gradients:
                    # conveniently, since we have the chunk probe, 
                    # mind as well rolling update the weighted average grad vector
                    # by accumulating the coeff * probe into the matching slice of grad_buffer
                    grad_buffer[param_idx][chunk_idx].add_(probe, alpha=grad_estimate)
                    
            

    # for idx, param in enumerate(model_params):
    #     # if idx == 3:                      # skip states tensor
    #     #     continue
    
    #     # -------- accumulate gradient in FP32 -------------------------
    #     grad = None
    #     for j in range(num_perturbations):
    #         seed         = all_seeds[j] + idx
    #         coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
    #         probe   = generate_perturbation(
    #                            param, coeff, distribution, seed)
            
    #         grad = probe if grad is None else grad + probe

    #     param.data.add_(grad, alpha= -1./float(num_perturbations)   )
    #     # -------- SGD step in FP32, then cast back -------------------
    #     # param_fp32 = param.data.float()
    #     # param_fp32.add_( -grad )                          # **NO extra ε/n**
    #     # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16
    # # SGD update (same as before)
    # param.data.add_(grad, alpha=-1. / float(num_perturbations))

    if cache_gradients:
        
        # for buf, param in zip(grad_buffer, model_params):
            # param.data.add_(buf) # buf already stores −∇̂θ / n
        # for i, (g, param) in enumerate(zip(grad_buffer, model_params)):
        for param_idx, param in enumerate(model_params):
            for chunk_idx, chunk in enumerate(torch.split(param.data, CHUNK_SIZE, dim=0)):
                
                g = grad_buffer[param_idx][chunk_idx]
    
                # Apply RMSProp and/or momentum 
                if args.beta1>0:
                    momentum_i = args.coordinate_momentum[param_idx][chunk_idx]
                    momentum_i.mul_(args.beta1).add_(g, alpha=1 - args.beta1)   # m_t
                    m_hat = momentum_i   
                    if np.random.rand()>0.99:
                        print("----- m_hat:")
                        print(m_hat[:20])
                else:
                    m_hat = g
                
                if args.beta2>0:
                    variances_i = args.coordinate_variance[param_idx][chunk_idx]
                    variances_i.mul_(args.beta2).addcmul_(g, g, value=1 - args.beta2)  # v_t
                    v_hat = variances_i.sqrt().add_(1e-8)            # ε for numerical stability
                    if np.random.rand()>0.99:
                        print("----- v_hat:")
                        print(v_hat[:20])
                else:
                    v_hat = 1.0# Parameter update


                if args.use_probe_preconditioning:
                    chunk.data.add_( lr_to_eta_ratio * m_hat ) #  v_hat is already in the chunk.
                else:
                    chunk.data.add_( lr_to_eta_ratio * m_hat / v_hat )
    
                # ---- Decoupled weight decay (AdamW style) --------------------
                if args.weight_decay > 0.0:
                    # Skip 1-D tensors (biases, LayerNorm/BatchNorm weights)
                    if param.ndim > 1:
                        chunk.add_(chunk, alpha = -lr * args.weight_decay)
    

    

    else: # calculate this post from the seeds
        raise Exception("This branch is no longer maintained. Need to implement momentum and var and weight decay")
        
        # for idx, param in enumerate(model_params):
        #     # grad = None
        #     for j in range(num_perturbations):
        #         seed  = all_seeds[j] + idx
        #         coeff = grad_estimate_list[j]                 # (f⁺−f⁻)/2n # no ε bc ε == lr and they cancel out
        
        #         # ── NEW: accumulate probe chunk-by-chunk ─────────────────────
        #         for chunk in torch.split(param, CHUNK_SIZE, dim=0):     # 8k rows
        #             probe = generate_perturbation(
        #                         chunk, lr_to_eta_ratio * coeff, distribution, seed)
        #             chunk.add_(probe) 
        
        #         # grad = probe if grad is None else grad + probe


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = sum_of_losses/num_perturbations
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics


def cdrge_no_chunking(model_params, loss_fn, lr, epsilon, iterations, num_perturbations=20,
                   distribution='rad', antithetic=False, use_adaptive_step=False, clip_grad_norm=0.0,
                   args=None):
    """DO NOT USE ... LIKELY: Central Difference Random Gradient Estimation optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Current loss at start of iteration
    if is_verbose:
        print("CDRGE: Computing initial loss...")
    
    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    gradients = [torch.zeros_like(p.data) for p in model_params]


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0
    if is_verbose:
        print("starting perts")
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = (pos_loss - neg_loss) / (2) # no epsilon bc we mult later so it cancels out
        sum_of_losses += (pos_loss + neg_loss) / (2)
        grad_estimate_list.append(grad_estimate)
        
        for param_idx, param in enumerate(model_params):
            # Regenerate the same noise
            probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)

            # Restore parameter by adding the perturbation
            param.data.add_(probe)
            

    for idx, param in enumerate(model_params):
        # if idx == 3:                      # skip states tensor
        #     continue
    
        # -------- accumulate gradient in FP32 -------------------------
        grad = None
        for j in range(num_perturbations):
            seed         = all_seeds[j] + idx
            coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
            probe   = generate_perturbation(
                               param, coeff, distribution, seed)
            
            grad = probe if grad is None else grad + probe

        param.data.add_(grad, alpha= -1./float(num_perturbations)   )
        # -------- SGD step in FP32, then cast back -------------------
        # param_fp32 = param.data.float()
        # param_fp32.add_( -grad )                          # **NO extra ε/n**
        # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = sum_of_losses/num_perturbations
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics


def fdras_optimize(model_params, loss_fn, lr, epsilon, iterations, num_perturbations=20,
                   distribution='rad', antithetic=False, use_adaptive_step=False, clip_grad_norm=0.0,
                   args=None):
    """Finite Difference Random Adaptive Sampling (FDRAS) optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"FDRAS: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"FDRAS: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Current loss at start of iteration
    if is_verbose:
        print("FDRAS: Computing initial loss...")
    
    # Compute initial loss as average across all macro batches
    total_clean_loss = 0.0
    for _ in range(macro_batch_size):
        clean_loss_batch = loss_fn()  # Current model state loss
        batch_loss = clean_loss_batch.item() if hasattr(clean_loss_batch, 'item') else float(clean_loss_batch)
        total_clean_loss += batch_loss
        
    clean_loss = total_clean_loss / macro_batch_size
    
    if is_verbose:
        print(f"FDRAS: Initial loss: {clean_loss:.6f}")

    # Store gradients for each parameter
    if is_verbose:
        print("FDRAS: Initializing gradients...")
    gradients = [torch.zeros_like(p.data) for p in model_params]

    antithetic_mult = 1
    
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = random.randint(0, 2**32 - 1)

        if antithetic and j%2==1:
            antithetic_mult = -1
            
        # Apply positive perturbations using the seed
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, antithetic_mult*epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size
        
        # Single-sided estimation: (f(x+ε) - f(x))/ε
        grad_estimate = (pos_loss - clean_loss) / epsilon

        for param_idx, param in enumerate(model_params):
            # Reset seed to get the same perturbation for restoring

            # Regenerate the same noise
            probe = generate_perturbation(param, antithetic_mult, distribution, seed=pert_seed + param_idx)

            # Update gradient accumulation - avoid division by zero
            gradients[param_idx].add_(probe * grad_estimate)

            # Restore parameter by negating the perturbation
            param.data.add_(-probe*epsilon)

    # Average gradients across perturbations
    for grad in gradients:
        grad.div_(num_perturbations) # not actual_perturbs

    # Compute total gradient norm
    # total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients]))

    # Use a default clip_norm if none provided
    # effective_clip_norm = 10.0 if clip_grad_norm <= 0 else clip_grad_norm

    # # Apply clipping if gradient norm is too large
    # if total_norm > effective_clip_norm:
    #     clip_coef = effective_clip_norm / (total_norm + 1e-8)
    #     for g in gradients:
    #         g.mul_(clip_coef)

    # Store gradient norm for metrics
    grad_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients])).item()
    metrics['grad_norm'].append(grad_norm)

    # Default step size
    current_lr = lr

    # if use_adaptive_step:
    #     # Use adaptive step sizes only if loss doesn't decrease enough
    #     # Try different step sizes and keep the best
    #     loss_decreased = False

    #     # Standard step sizes to try
    #     step_sizes = [lr * 0.1, lr * 0.5, lr, lr * 2.0, lr * 5.0]

    #     best_step_loss = float('inf')
    #     best_step_lr = lr

    #     # Save current parameters - more efficient approach using list comprehension
    #     orig_params_values = [p.data.clone() for p in model_params]

    #     for step_size in step_sizes:
    #         # Apply gradient with this step size
    #         for param_idx, (param, grad) in enumerate(zip(model_params, gradients)):
    #             # Start from clean copy
    #             param.data.copy_(orig_params_values[param_idx])
    #             # Apply gradient (subtract because we want to minimize)
    #             param.data.add_(grad, alpha=-step_size)

    #         # Evaluate parameters
    #         step_loss_val = loss_fn()
    #         step_loss = step_loss_val.item() if hasattr(step_loss_val, 'item') else float(step_loss_val)

    #         # Loss decreased significantly
    #         if step_loss < (clean_loss - 0.001):
    #             loss_decreased = True

    #         # Keep track of best step size
    #         if step_loss < best_step_loss:
    #             best_step_loss = step_loss
    #             best_step_lr = step_size

    #     # If loss decreased with any step size, use that step size
    #     if loss_decreased:
    #         current_lr = best_step_lr

    #     # Restore original parameters before applying the best step size
    #     for param_idx, param in enumerate(model_params):
    #         param.data.copy_(orig_params_values[param_idx])

    # Apply gradient with the chosen step size (simple SGD)
    for param_idx, param in enumerate(model_params):
        param.data.add_(gradients[param_idx], alpha=-current_lr)

    # Check if loss improved with the update
    # new_loss_val = loss_fn()
    # new_loss = new_loss_val.item() if hasattr(new_loss_val, 'item') else float(new_loss_val)

    # Reverse out the update if it made us worse? Maybe we should just let it do its thing..
    # if new_loss > best_loss:
    #     # Restore best parameters if we've seen better ones
    #     print("NOT BETTER!")
    #     for param_idx, (param, grad) in enumerate(zip(model_params, gradients)):
    #         param.data.add_(grad, alpha=current_lr)

    # Track metrics
    elapsed = time.time() - start_time
    metrics['train_loss'].append(clean_loss)
    metrics['elapsed_time'].append(elapsed)


    # print(f"FDRAS: Optimization completed in {elapsed:.2f}s with loss {clean_loss:.6f}")

    # # Log progress
    # if iterations <= 10 or (iter_i + 1) % max(1, iterations // 10) == 0:
    #     print(f"Iter {iter_i+1}/{iterations}: Loss {clean_loss:.4f} (time: {elapsed:.2f}s, grad_norm: {grad_norm:.4f})")

    return metrics





import torch, time, random
from typing import List

# ────────────────────────────────────────────────────────────────────────────────
# Second‑Order SPSA (2SPSA) Solver
# ────────────────────────────────────────────────────────────────────────────────
def SPSA2(model_params: List[torch.Tensor],
          loss_fn,
          lr: float,
          epsilon: float,
          iterations: int,
          num_perturbations: int = 20,
          distribution: str = 'rad',
          antithetic: bool = False,        # kept for interface parity (ignored)
          use_adaptive_step: bool = False, # kept for interface parity (ignored)
          clip_grad_norm: float = 0.0,
          cache_gradients: bool = False,
          lambda_reg: float = 1.0,
          CHUNK_SIZE: int = 2 ** 15,
          args=None):
    """
    Second‑Order SPSA (2SPSA) Zeroth‑Order Optimizer

    Implements the Newton‑style update

        Δθ = (1 / 2n) Σ_i  [(f(θ+εv_i) - f(θ−εv_i)) v_i] /
                           { [(f(θ+εv_i+εu_i) − f(θ+εv_i−εu_i)
                              −f(θ−εv_i+εu_i) + f(θ−εv_i−εu_i))
                             (v_iᵀu_i) / (4 ε²)]  + λ_reg }

    using *n* independent probe pairs (v_i , u_i), each drawn from the
    Rademacher distribution.  Six forward passes are therefore required
    per probe:

        θ ± εv_i ,
        θ + εv_i ± εu_i ,
        θ − εv_i ± εu_i .

    All memory‑saving tricks from SPSA1/1.5 are retained: parameters are
    perturbed in‑place chunk‑by‑chunk, a seeded RNG reproduces the probes,
    and only per‑probe scalar coefficients are kept.
    """
    # ───── House‑keeping & logging ────────────────────────────────────────────
    is_verbose = args is not None and getattr(args, 'verbose', False)
    if is_verbose:
        print(f"2SPSA: Starting optimisation with {num_perturbations} perturbations")

    metrics = {'iterations': [], 'train_loss': [], 'elapsed_time': [], 'grad_norm': []}
    device   = model_params[0].device
    mb_size  = getattr(args, 'macro_batch_size', 1)

    # ───── Helpers ────────────────────────────────────────────────────────────
    OFFSET_U = 2 ** 20           # deterministic offset so u_seeds != v_seeds

    def _apply_noise(scale_v: float, scale_u: float,
                     seeds_v, seeds_u):
        """
        Add (scale_v * v  +  scale_u * u) to the parameters in‑place.
        Both scales are real numbers (can be ±epsilon, ±2*epsilon, etc.).
        """
        for p_idx, p in enumerate(model_params):
            for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
                if scale_v:
                    v_probe = generate_perturbation(
                        chunk, scale_v, distribution, seed=seeds_v[p_idx])
                    chunk.add_(v_probe)
                if scale_u:
                    u_probe = generate_perturbation(
                        chunk, scale_u, distribution, seed=seeds_u[p_idx])
                    chunk.add_(u_probe)

    def _eval_offset(scale_v: float, scale_u: float,
                     seeds_v, seeds_u) -> float:
        """
        Apply the requested offset, compute (macro‑)batched loss, then revert.
        Returns the averaged scalar loss value.
        """
        _apply_noise(scale_v, scale_u, seeds_v, seeds_u)

        tot = 0.0
        for _ in range(mb_size):
            l = loss_fn()
            tot += float(l.item() if hasattr(l, 'item') else l)

        _apply_noise(-scale_v, -scale_u, seeds_v, seeds_u)  # revert
        return tot / mb_size

    # ───── Pre‑compute "clean" loss ───────────────────────────────────────────
    clean_loss = sum(float(loss_fn().item()) for _ in range(mb_size)) / mb_size

    # ───── Storage for per‑probe scalar coefficients ‑‑ one per perturbation ‑─
    grad_coeffs: List[float] = []
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_perturbations)]

    # ───── Begin optimisation ‑‑ one 2SPSA probe at a time ────────────────────
    t0 = time.time()
    for j in range(num_perturbations):

        base_seed = seeds_list[j]
        # Per‑parameter deterministic seeds for v and u
        seeds_v = [base_seed + idx          for idx in range(len(model_params))]
        seeds_u = [base_seed + OFFSET_U + idx for idx in range(len(model_params))]

        # ─── Step 1 & 2: θ ± εv_i (gradient numerator pieces) ───────────────
        pos_v = _eval_offset(+epsilon, 0.0, seeds_v, seeds_u)
        neg_v = _eval_offset(-epsilon, 0.0, seeds_v, seeds_u)

        # ─── Steps 3–6: Hessian‑like denominator pieces ─────────────────────
        pos_v_pos_u = _eval_offset(+epsilon, +epsilon, seeds_v, seeds_u)
        pos_v_neg_u = _eval_offset(+epsilon, -epsilon, seeds_v, seeds_u)
        neg_v_pos_u = _eval_offset(-epsilon, +epsilon, seeds_v, seeds_u)
        neg_v_neg_u = _eval_offset(-epsilon, -epsilon, seeds_v, seeds_u)

        # ─── Compute vᵀu for this probe (needs only signs, so scale 1) ──────
        # vTu = 0.0
        # for p_idx, p in enumerate(model_params):
        #     for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
        #         v_signs = generate_perturbation(chunk, 1.0, distribution,
        #                                         seed=seeds_v[p_idx])
        #         u_signs = generate_perturbation(chunk, 1.0, distribution,
        #                                         seed=seeds_u[p_idx])
        #         vTu += float((v_signs * u_signs).sum().item())

        # ─── Assemble numerator & denominator ‑‑ equation (C) ───────────────
        num  =  (pos_v - neg_v)                      # scalar
        # curv = ((pos_v_pos_u - pos_v_neg_u
        #         -neg_v_pos_u + neg_v_neg_u) * vTu) / (4.0 * epsilon ** 2)

        # denom = max(curv, lambda_reg)
        # coeff = - (num / denom) / (2.0 * num_perturbations)  # minus for descent

        raw_curv = ((pos_v_pos_u - pos_v_neg_u
            -neg_v_pos_u + neg_v_neg_u) ) / (4 * epsilon ** 2)

        #denom =  math.copysign(max(abs(raw_curv), lambda_reg), raw_curv)

        #denom = max( (raw_curv)**args.saturating_alpha, lambda_reg)

        if abs(raw_curv)<1.:
            print("raw_curve small")
            raw_curv = 1.
        denom = raw_curv
        
        coeff = -(pos_v - neg_v) / (2 * num_perturbations * denom)

        grad_coeffs.append(coeff)

        if True: #is_verbose:
            print(f"  pert: {j} "
                  f"num = {num:.8f}, "
                  f"coeff = {coeff:.8f}, "
                  f"curv = {raw_curv:.4f}, denom = {denom:.4f} "
                  f"pos_v={pos_v:.4f} neg_v={neg_v:.4f} "
                  f"pos_v_pos_u={pos_v_pos_u:.4f} pos_v_neg_u={pos_v_neg_u:.4f} "
                  f"neg_v_pos_u={neg_v_pos_u:.4f} neg_v_neg_u={neg_v_neg_u:.4f} " # vTu {vTu:.4f}
            )
            

    # ───── Apply the accumulated update (cached or regenerated) ──────────────
    if cache_gradients:
        grad_buffer = [torch.zeros_like(p.data) for p in model_params]

        for j, coeff in enumerate(grad_coeffs):
            seeds_v = [seeds_list[j] + idx for idx in range(len(model_params))]
            for p_idx, buf in enumerate(grad_buffer):
                for chunk_buf in torch.split(buf, CHUNK_SIZE, dim=0):
                    probe = generate_perturbation(chunk_buf, coeff, distribution,
                                                  seed=seeds_v[p_idx])
                    chunk_buf.add_(probe)

        for buf, p in zip(grad_buffer, model_params):
            p.data.add_(buf)

    else:
        for p_idx, p in enumerate(model_params):
            for j, coeff in enumerate(grad_coeffs):
                seed = seeds_list[j] + p_idx
                for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
                    probe = generate_perturbation(chunk, coeff, distribution, seed)
                    chunk.add_(probe)

    # ───── Final logging & metrics ───────────────────────────────────────────
    elapsed = time.time() - t0
    metrics['train_loss'].append(clean_loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"2SPSA: Completed in {elapsed:.2f}s ‑ clean loss {clean_loss:.6f}")

    return metrics




import pdb
from collections import deque

def BanditSPSA(model_params,
               loss_fn,
               lr,
               epsilon,
               iterations,                  # <-- ignored; keep for API parity
               num_perturbations=20,
               distribution='rad',
               antithetic=False,
               use_adaptive_step=False,
               clip_grad_norm=0.0,
               cache_gradients=True,
               CHUNK_SIZE=2**15,
               args=None):
    """
    Bandit-SPSA step (one optimiser update).
    Keeps the exact public interface of SPSA1.
    """

    print("BanditSPSA")
    # ────────────────────────────────────────────────────────────────────
    # 0.  persistent state (reservoir & step counter)                    │
    # ────────────────────────────────────────────────────────────────────
    if not hasattr(BanditSPSA, "_state"):
        BanditSPSA._state = {"step": 0,
                             "reservoir": {}  # seed ▸ {'ema':float,'pulls':int,'grads':deque}
                            }
    st: Dict = BanditSPSA._state
    step        = st["step"]
    reservoir   = st["reservoir"]            # type: Dict[int, Dict]

    # print("="*50)
    # print("Before")
    # print("="*50)
    # print(st)
    # print("="*50)
    
    # Hard reset every 1000 updates
    if step and step % 1000 == 0:
        reservoir.clear()

    # exploitation share pt(t) : 0.2 → 0.9 (logistic in first 500 steps)
    def pt_sched(t: int) -> float:
        return 0.20 + 0.70 / (1.0 + math.exp(-(t - 250) / 75.0))
    pt = pt_sched(step)

    # ────────────────────────────────────────────────────────────────────
    # 1.  choose seeds: exploit (reservoir) + explore (new)              │
    # ────────────────────────────────────────────────────────────────────
    n_exploit = min(int(round(num_perturbations * pt)), len(reservoir))
    n_explore = num_perturbations - n_exploit

    exploit_seeds, exploit_probs = [], []     # sampling probs inside reservoir
    if n_exploit:
        # UCB-Boltzmann scores
        total_pulls = sum(r['pulls'] for r in reservoir.values()) or 1
        seeds   = list(reservoir.keys())
        # scores  = [abs(r['ema']) +
        #            math.sqrt(math.log(total_pulls + 1.0) /
        #                      (r['pulls'] + 1e-9))
        #            for r in reservoir.values()]
        scores  = [abs(r['ema'])  for r in reservoir.values()] # NO UCB method

        m       = max(scores)
        softmax = [math.exp( (s - m)/args.bandit_softmax_temperature) for s in scores]
        Z       = sum(softmax)
        probs   = [s/Z for s in softmax]

        # sample WITHOUT replacement until n_exploit collected
        # (retry if duplicates happen due to replacement-by-default)
        while len(exploit_seeds) < n_exploit and seeds:
            choice = random.choices(seeds, weights=probs, k=1)[0]
            idx    = seeds.index(choice)
            exploit_seeds.append(choice)
            exploit_probs.append(probs[idx])
            seeds.pop(idx); probs.pop(idx)

    # ───────── VERBOSE BLOCK ────────────────────────────
    if True:
        print(f"Reservoir size {len(reservoir.items())}")
        # 1) Top-10 probes in the reservoir by |EMA|
        if reservoir:
            print("\n── Top-10 reservoir EMAs ──")
            for rank, (s, r) in enumerate(
                    sorted(reservoir.items(),
                           key=lambda kv: -abs(kv[1]['ema']))[:10], 1):
                print(f"{rank:2d}. seed={s:>10}  ema={r['ema']:+.3e}  "
                      f"pulls={r['pulls']:4d}  last_grad={r['grads'][-1][1]:+.2e}")
        else:
            print("\n[reservoir is empty]")
    
        # 2) UCB scores for the seeds we just chose to exploit
        if exploit_seeds:
            print("\n── Exploit selection this step ──")
            for s, p in zip(exploit_seeds, exploit_probs):
                r     = reservoir[s]
                # score = abs(r['ema']) + math.sqrt(
                #             math.log(sum(rr['pulls'] for rr in reservoir.values())+1)
                #             /(r['pulls']+1e-9))
                score = abs(r['ema']) # NO UCB
                print(f"seed={s:>10}  UCB={score:+.3e}  "
                      f"ema={r['ema']:+.3e}  pulls={r['pulls']:4d}  "
                      f"prob={p:.3f}")
        print("────────────────────────────────────\n")
        # pdb.set_trace()
    # ───────────────── END VERBOSE BLOCK ────────────────────────────────
    

    
    # exploration seeds (unique & not in reservoir)
    explore_seeds = []
    while len(explore_seeds) < n_explore:
        s = random.randint(0, 2**32 - 1)
        if s not in reservoir and s not in exploit_seeds:
            explore_seeds.append(s)

    # concat lists for main loop
    all_seeds  = exploit_seeds + explore_seeds
    all_probs  = exploit_probs + [None] * n_explore   # None ⇒ explore

    # per-seed importance weight  w = 1 / (num_perturbations * P(s))
    #   • exploit:   P(s) = pt * p_i
    #   • explore:   P(s) = (1-pt) / n_explore  (≈ uniform over chosen newcomers)
    weights = []
    for p in all_probs:
        if p is None:  # explore
            if n_explore == 0 or (1-pt) < 1e-12:
                weights.append(1.0)           # degenerate but safe
            else:
                weights.append(1.0)
                # weights.append(1.0 / (num_perturbations * (1-pt) / n_explore))
        else:           # exploit
            weights.append(1.0)
            # weights.append(1.0 / (num_perturbations * pt * p))

    # ────────────────────────────────────────────────────────────────────
    # 2.  boiler-plate from SPSA1  (buffers, momentum, variance)         │
    # ────────────────────────────────────────────────────────────────────
    is_verbose = bool(args and getattr(args, "verbose", False))
    lr_eta     = lr / epsilon
    macro_bs   = getattr(args, "macro_batch_size", 1)
    start_t    = time.time()

    if getattr(args, "beta1", 0.0) > 0:
        coord_mom = [list(torch.split(torch.zeros_like(p.data), CHUNK_SIZE, dim=0))
                     for p in model_params]
    if getattr(args, "beta2", 0.0) > 0:
        coord_var = [list(torch.split(torch.ones_like(p.data), CHUNK_SIZE, dim=0))
                     for p in model_params]
    if (getattr(args, "beta1", 0.0) > 0 or getattr(args, "beta2", 0.0) > 0):
        cache_gradients = True
    if cache_gradients:
        grad_buf = [list(torch.split(torch.zeros_like(p.data), CHUNK_SIZE, dim=0))
                    for p in model_params]
    else:
        raise RuntimeError("BanditSPSA requires cache_gradients=True")

    sum_loss = 0.0

    # ────────────────────────────────────────────────────────────────────
    # 3.  loop over probes                                              │
    # ────────────────────────────────────────────────────────────────────
    for seed, w in zip(all_seeds, weights):
        # local torch.Generator to avoid clobbering global RNG
        # ----------------------------------------------------
        def gen_probe(chunk, scale):
            g = torch.Generator(device=chunk.device); g.manual_seed(seed + scale)
            return generate_perturbation(chunk, scale, distribution, seed + scale)

        # +ε
        for p_idx, p in enumerate(model_params):
            for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
                probe = generate_perturbation(
                    ch, 1, distribution, seed + p_idx + c_idx)
                if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
                    probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
                ch.add_(probe, alpha=epsilon)

        pos = sum(float(loss_fn().item() if hasattr(loss_fn(), "item") else loss_fn())
                  for _ in range(macro_bs)) / macro_bs

        # –ε  (add −2ε probe)
        for p_idx, p in enumerate(model_params):
            for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
                probe = generate_perturbation(
                    ch, -2, distribution, seed + p_idx + c_idx)
                if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
                    probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
                ch.add_(probe, alpha=epsilon)

        neg = sum(float(loss_fn().item() if hasattr(loss_fn(), "item") else loss_fn())
                  for _ in range(macro_bs)) / macro_bs

        raw_fd   = -(pos - neg) / 2.0                     # finite diff
        coeff    = raw_fd * w                             # IS-weighted coeff
        sum_loss += (pos + neg) / 2.0

        # restore params & accumulate gradient
        for p_idx, p in enumerate(model_params):
            for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
                probe = generate_perturbation(
                    ch, 1, distribution, seed + p_idx + c_idx)
                if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
                    probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
                ch.add_(probe, alpha=epsilon)             # back to θ
                grad_buf[p_idx][c_idx].add_(probe, alpha=coeff)

        # ─ update / insert into reservoir ────────────────────────────
        if abs(raw_fd) > 1e-3:
            rec = reservoir.get(seed)
            if rec is None:
                reservoir[seed] = {'ema': raw_fd,
                                   'pulls': 1,
                                   'grads': deque([(step, raw_fd)], maxlen=20)}
            else:
                rec['pulls'] += 1
                rec['ema']  = 0.5 * rec['ema'] + 0.5 * raw_fd
                rec['grads'].append((step, raw_fd))

            # prune if >10 k
            if len(reservoir) > 10_000:
                for s, _ in sorted(reservoir.items(),
                                   key=lambda kv: abs(kv[1]['ema']))[:len(reservoir)-10_000]:
                    reservoir.pop(s, None)

    # ────────────────────────────────────────────────────────────────────
    # 4.  parameter update                                              │
    # ────────────────────────────────────────────────────────────────────
    for p_idx, p in enumerate(model_params):
        for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
            g = grad_buf[p_idx][c_idx]

            if getattr(args, "beta1", 0.0) > 0:
                m = coord_mom[p_idx][c_idx]
                m.mul_(args.beta1).add_(g, alpha=1 - args.beta1)
                m_hat = m
            else:
                m_hat = g

            if getattr(args, "beta2", 0.0) > 0:
                v = coord_var[p_idx][c_idx]
                v.mul_(args.beta2).addcmul_(g, g, value=1 - args.beta2)
                v_hat = v.sqrt().add_(1e-8)
            else:
                v_hat = 1.0

            ch.add_(lr_eta * m_hat / v_hat)

            if getattr(args, "weight_decay", 0.0) > 0.0 and p.ndim > 1:
                ch.add_(ch, alpha=-lr * args.weight_decay)

    # ────────────────────────────────────────────────────────────────────
    # 5.  bookkeeping                                                   │
    # ────────────────────────────────────────────────────────────────────
    if is_verbose:
        el  = time.time() - start_t
        print(f"[BanditSPSA] step={step}  pt={pt:.3f}  "
              f"loss={sum_loss/num_perturbations:.6f}  "
              f"reservoir={len(reservoir)}  Δt={el:.2f}s")

    st["step"] += 1

    return {
        "iterations": [step],
        "train_loss": [sum_loss / num_perturbations],
        "elapsed_time": [time.time() - start_t],
        "grad_norm": []
    }


# def SangerSPSA(model_params,
#            loss_fn,
#            lr,
#            epsilon,
#            iterations,                    # kept for API parity (unused)
#            num_perturbations=20,
#            distribution='rad',
#            antithetic=False,
#            use_adaptive_step=False,
#            clip_grad_norm=0.0,
#            cache_gradients=True,
#            CHUNK_SIZE=2**15,
#            args=None):
#     """
#     One optimisation step of Sub-space Pre-conditioned SPSA (Sanger-SPSA).
#     Interface and coding patterns follow SPSA1 exactly.
#     """

#     # ────────────────────────────────────────────────────────────────────
#     # 0.  persistent state for the sub-space                            │
#     # ────────────────────────────────────────────────────────────────────
#     rank = args.sanger_rank  
#     warmup_iters = args.warmup_iters
#     base_lr = 1e-4
    
#     # WARMUP THE LR SO YOU GIVE THE SUBSPACE A CHANCE TO LEARN ANYTHING
#     # update lr warmup factor
#     warmup_factor = min(iterations / float(warmup_iters), 1.0)
    
#     # update LR for this step
#     learning_rate = base_lr + args.learning_rate * warmup_factor
    
#     beta_eigen_sanger = base_lr + args.beta_eigen_sanger - args.beta_eigen_sanger * warmup_factor
    
#     dim       = sum(p.numel() for p in model_params)
#     device    = model_params[0].device
#     dtype     = model_params[0].dtype
    
#     if not hasattr(args, "_state"):
#         print("initializing Sanger V")
#         V = torch.empty(dim, rank, device=device, dtype=torch.float32).normal_()
#         V, _ = torch.linalg.qr(V, mode="reduced")          # orthonormal fp32 columns
    
#         args._state = {
#             "step": 0,
#             "V"  : V,                                      # (d × n)  fp32
#             "lam": torch.ones(rank, device=device, dtype=torch.float32)
#         }
#         print("done initializing Sanger V")


#     st   = args._state
#     V    = st["V"]                                       # (d × n)
#     lam  = st["lam"]                                     # (n,)
#     step = st["step"]

#     # ────────────────────────────────────────────────────────────────────
#     # 1.  build SPSA gradient estimate  (identical to SPSA1 core)       │
#     # ────────────────────────────────────────────────────────────────────
#     lr_eta     = learning_rate / epsilon
#     macro_bs   = getattr(args, "macro_batch_size", 1)
#     is_verbose = bool(args and getattr(args, "verbose", False))
#     start_t    = time.time()

#     if getattr(args, "beta1", 0.0) > 0:
#         coord_mom = [list(torch.split(torch.zeros_like(p.data), CHUNK_SIZE, dim=0))
#                      for p in model_params]
#     if getattr(args, "beta2", 0.0) > 0:
#         coord_var = [list(torch.split(torch.ones_like(p.data), CHUNK_SIZE, dim=0))
#                      for p in model_params]
#     if (getattr(args, "beta1", 0.0) > 0 or getattr(args, "beta2", 0.0) > 0):
#         cache_gradients = True
#     if cache_gradients:
#         grad_buf = [list(torch.split(torch.zeros_like(p.data), CHUNK_SIZE, dim=0))
#                     for p in model_params]
#     else:
#         raise RuntimeError("SangerSPSA requires cache_gradients=True")

#     sum_loss = 0.0
#     all_seeds: List[int] = [random.randint(0, 2**32 - 1) for _ in range(num_perturbations)]

#     for seed in all_seeds:
#         # +ε
#         for p_idx, p in enumerate(model_params):
#             for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
#                 probe = generate_perturbation(
#                     ch, 1, distribution, seed + p_idx + c_idx)
#                 if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
#                     probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
#                 ch.add_(probe, alpha=epsilon)

#         pos = sum(float(loss_fn().item() if hasattr(loss_fn(), "item") else loss_fn())
#                   for _ in range(macro_bs)) / macro_bs

#         # –ε
#         for p_idx, p in enumerate(model_params):
#             for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
#                 probe = generate_perturbation(
#                     ch, -2, distribution, seed + p_idx + c_idx)
#                 if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
#                     probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
#                 ch.add_(probe, alpha=epsilon)

#         neg = sum(float(loss_fn().item() if hasattr(loss_fn(), "item") else loss_fn())
#                   for _ in range(macro_bs)) / macro_bs

#         fd_coeff = -(pos - neg) / (2 * num_perturbations)   # 1-sided scaling handled here
#         sum_loss += (pos + neg) / 2.0

#         # restore θ and accumulate gradient
#         for p_idx, p in enumerate(model_params):
#             for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
#                 probe = generate_perturbation(
#                     ch, 1, distribution, seed + p_idx + c_idx)
#                 if getattr(args, "beta2", 0.0) and args.use_probe_preconditioning:
#                     probe /= torch.clamp_min(coord_var[p_idx][c_idx], 1.0)
#                 ch.add_(probe, alpha=epsilon)
#                 grad_buf[p_idx][c_idx].add_(probe, alpha=fd_coeff)

#     # flatten accumulated gradient  g  (d-vector, fp32 for maths)
#     g_flat = torch.cat(
#         [torch.cat([c.reshape(-1) for c in chunks]) for chunks in grad_buf]
#     ).to(dtype=torch.float32)

#     # ────────────────────────────────────────────────────────────────────
#     # 2.  Sanger-SPSA pre-conditioning                                       │
#     #     P g = (I-VVᵀ)g  +  V Λ⁻¹ Vᵀ g                                   │
#     # ────────────────────────────────────────────────────────────────────
#     g_proj      = V.T @ g_flat                       # (n,)
#     pre_g_flat = V @ (g_proj / lam)                 # whitened/parallel part
#     # pre_g_flat  += g_flat - V @ g_proj                # orthogonal part (noise?)
    
#     # distribute pre_g_flat back into param-shaped chunks & update θ
#     offset = 0
#     for p_idx, p in enumerate(model_params):
#         for c_idx, ch in enumerate(torch.split(p.data, CHUNK_SIZE, dim=0)):
#             numel = ch.numel()
#             g_chunk = pre_g_flat[offset:offset+numel].view_as(ch)
#             offset += numel

#             # momentum / RMSProp
#             if getattr(args, "beta1", 0.0) > 0:
#                 m = coord_mom[p_idx][c_idx]
#                 m.mul_(args.beta1).add_(g_chunk, alpha=1 - args.beta1)
#                 m_hat = m
#             else:
#                 m_hat = g_chunk

#             if getattr(args, "beta2", 0.0) > 0:
#                 v = coord_var[p_idx][c_idx]
#                 v.mul_(args.beta2).addcmul_(g_chunk, g_chunk, value=1 - args.beta2)
#                 v_hat = v.sqrt().add_(1e-8)
#             else:
#                 v_hat = 1.0

#             ch.add_(lr_eta * m_hat / v_hat)          # Robbins–Monro step
#             if getattr(args, "weight_decay", 0.0) > 0.0 and p.ndim > 1:
#                 ch.add_(ch, alpha=-lr * args.weight_decay)

#     # ────────────────────────────────────────────────────────────────────
#     # 3.  Sanger/Oja update of sub-space                                 │
#     # ────────────────────────────────────────────────────────────────────
#     # print("updating Sanger values")
#     with torch.no_grad():
#         # g_flat = g_flat / (g_flat.norm() + 1e-12)    # optional normalisation
#         g_proj = V.T @ g_flat                        # (n,)
#         # Sanger rule
#         V += beta_eigen_sanger * (g_flat.view(-1, 1) @ g_proj.view(1, -1)
#                     - V @ torch.triu(g_proj.view(-1, 1) @ g_proj.view(1, -1)))
#         # renormalise columns periodically
#         if (step + 1) % args.sanger_qr_every == 0:
#             V, _ = torch.linalg.qr(V, mode="reduced")        # orthonormal columns
#             # V = torch.nn.functional.normalize(V, dim=0)   # each column ‖v‖₂ = 1


#         # eigen-value EWMA
#         lam.mul_(1 - beta_eigen_sanger).add_(g_proj.pow(2), alpha=beta_eigen_sanger).clamp_(min=1e-8)
#         if np.random.rand()>0.99:
#             print("--lambdas:")
#             print(lam)
#         explained_var = g_proj.pow(2).sum() / g_flat.pow(2).sum()
#         print(f"Subspace captures {explained_var:.5%} of gradient variance")


#     # stash back
#     st["V"], st["lam"], st["step"] = V, lam, step + 1


#     # print("done updating Sanger values")
#     # ────────────────────────────────────────────────────────────────────
#     # 4.  bookkeeping                                                   │
#     # ────────────────────────────────────────────────────────────────────
#     if is_verbose:
#         dt = time.time() - start_t
#         print(f"[SangerSPSA] step={step:4d}  loss={sum_loss/num_perturbations:.6f} "
#               f"|V|={rank}  minλ={lam.min():.3e}  max|g_proj|={g_proj.abs().max():.3e}  Δt={dt:.2f}s")

#     return {
#         "iterations": [step],
#         "train_loss": [sum_loss / num_perturbations],
#         "elapsed_time": [time.time() - start_t],
#         "grad_norm": []    # (optional: compute if you need it)
#     }






# second try. got 1 working thats all.. 
# def SangerSPSA(model_params,
#            loss_fn,
#            lr,
#            epsilon,
#            iterations,                    # kept for API parity (unused)
#            num_perturbations=20,
#            distribution='rad',
#            antithetic=False,
#            use_adaptive_step=False,
#            clip_grad_norm=0.0,
#            cache_gradients=True,
#            CHUNK_SIZE=2**15,              # kept for API parity (ignored now)
#            args=None):
#     """
#     One optimisation step of Sub-space Pre-conditioned SPSA (Sanger-SPSA).

#     Differences vs. earlier version:
#         • No chunking / per-parameter loops – operate on the full flattened vector.
#         • Probe whitening:  probe ← V Λ⁻¹ Vᵀ probe  when args.use_probe_preconditioning.
#         • coord_mom / coord_var removed entirely.
#     """

#     # ─────────────────────────────────────────────────────────────────
#     # 0.  persistent state for the sub-space
#     # ─────────────────────────────────────────────────────────────────
#     rank          = args.sanger_rank
#     warmup_iters  = args.warmup_iters
#     base_lr       = 1e-4

#     warmup_factor = min(iterations / float(warmup_iters), 1.0)
#     learning_rate = base_lr + args.learning_rate * warmup_factor
#     beta_eigen_sanger = base_lr + args.beta_eigen_sanger - args.beta_eigen_sanger * warmup_factor

#     full_vec   = torch.nn.utils.parameters_to_vector(model_params).detach()
#     dim        = full_vec.numel()
#     device     = full_vec.device

#     if not hasattr(args, "_state"):
#         V = torch.randn(dim, rank, device=device, dtype=torch.float32)
#         V, _ = torch.linalg.qr(V, mode="reduced")       # orthonormal columns
#         args._state = dict(step=0,
#                            V=V,
#                            lam=torch.ones(rank, device=device, dtype=torch.float32))

#     st   = args._state
#     V    = st["V"]
#     lam  = st["lam"]
#     step = st["step"]

#     def precond(vec_f32):
#         proj = V.T @ vec_f32
#         return V @ (proj / (lam) )

#     lr_eta   = learning_rate / epsilon
#     macro_bs = getattr(args, "macro_batch_size", 1)
#     is_verbose = bool(args and getattr(args, "verbose", False))
#     start_t  = time.time()

#     grad_acc = torch.zeros_like(full_vec, dtype=torch.float32)

#     # ─────────────────────────────────────────────────────────────────
#     # 1.  build SPSA gradient estimate
#     # ─────────────────────────────────────────────────────────────────
#     seeds = [random.randint(0, 2**32 - 1) for _ in range(num_perturbations)]

#     for sd in seeds:
#         # +ε
#         probe = generate_perturbation(full_vec, 1, distribution, sd).to(torch.float32)
#         if args.use_probe_preconditioning:
#             probe = precond(probe)
#         full_vec.add_(probe, alpha=epsilon)
#         torch.nn.utils.vector_to_parameters(full_vec, model_params)

#         pos = sum(float(loss_fn()) for _ in range(macro_bs)) / macro_bs

#         # –ε
#         full_vec.add_(probe, alpha=-2 * epsilon)
#         torch.nn.utils.vector_to_parameters(full_vec, model_params)

#         neg = sum(float(loss_fn()) for _ in range(macro_bs)) / macro_bs

#         fd_coeff = -(pos - neg) / (2 * num_perturbations)
#         grad_acc.add_(probe, alpha=fd_coeff)

#         # restore θ
#         full_vec.add_(probe, alpha=epsilon)

#     torch.nn.utils.vector_to_parameters(full_vec, model_params)  # ensure params restored
#     g_flat = grad_acc

#     # ─────────────────────────────────────────────────────────────────
#     # 2.  low-rank Newton pre-conditioning:  Pg = V Λ⁻¹ Vᵀ g
#     # ─────────────────────────────────────────────────────────────────
#     pre_g = precond(g_flat)

#     full_vec.add_(pre_g, alpha=lr_eta)
#     torch.nn.utils.vector_to_parameters(full_vec, model_params)

#     # ─────────────────────────────────────────────────────────────────
#     # 3.  Sanger / Oja update of sub-space
#     # ─────────────────────────────────────────────────────────────────
#     with torch.no_grad():
#         g_proj = V.T @ g_flat
#         V += beta_eigen_sanger * (
#                  g_flat.unsqueeze(1) @ g_proj.unsqueeze(0)
#                - V @ torch.triu(g_proj.unsqueeze(1) @ g_proj.unsqueeze(0))
#         )
#         if (step + 1) % args.sanger_qr_every == 0:
#             V, _ = torch.linalg.qr(V, mode="reduced")

#         # lam.mul_(1 - beta_eigen_sanger).add_(g_proj.pow(2),
#         #                                      alpha=beta_eigen_sanger).clamp_(min=1e-8)

#         explained = (g_proj.pow(2).sum() / g_flat.pow(2).sum()).item()
#         print(f"Subspace captures {explained:.5%} of gradient variance")

#     st["V"], st["lam"], st["step"] = V, lam, step + 1

#     # ─────────────────────────────────────────────────────────────────
#     # 4.  bookkeeping
#     # ─────────────────────────────────────────────────────────────────
#     if is_verbose:
#         dt = time.time() - start_t
#         print(f"[SangerSPSA] step={step:4d} "
#               f"loss={(pos+neg)/2.0:.6f} |V|={rank} "
#               f"minλ={lam.min():.3e} max|g_proj|={g_proj.abs().max():.3e} "
#               f"Δt={dt:.2f}s")

#     return dict(iterations=[step],
#                 train_loss=[(pos + neg) / 2.0],
#                 elapsed_time=[time.time() - start_t],
#                 grad_norm=[])



def SangerSPSA(model_params,
               loss_fn,
               lr,
               epsilon,
               iterations,                    # kept for API parity
               num_perturbations=20,
               distribution="rad",
               antithetic=False,
               use_adaptive_step=False,
               clip_grad_norm=0.0,
               cache_gradients=True,
               CHUNK_SIZE=2**15,              # ignored
               args=None):
    """
    Sub-space Pre-conditioned SPSA with a single matrix W (P = W Wᵀ + αI).
    Memory-efficient column-wise W-update; no explicit d×n outer product.
    """
    import time, random, math
    import torch
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    # ───── 0. hyper-params & persistent state ─────
    rank         = args.sanger_rank
    base_lr      = 1e-4
    warmup_iters = args.warmup_iters
    alpha_eye_scalar  = args.alpha_eye_scalar
    β_eig_base   = args.beta_eigen_sanger

    warm     = min(float(iterations) / float(max(1, warmup_iters)), 1.0)
    lr_sched = base_lr + args.learning_rate * warm
    β_eig    = β_eig_base #base_lr + β_eig_base - β_eig_base * warm

    model_params_flat = parameters_to_vector(model_params).detach().to(torch.float32)
    d, dev = model_params_flat.numel(), model_params_flat.device

    if not hasattr(args, "_state"):
        # init W; keep your original QR init when possible
        W = torch.randn(d, rank, device=dev, dtype=torch.float32)
        try:
            W, _ = torch.linalg.qr(W, mode="reduced")
        except RuntimeError:
            # if QR is too big, just column-normalize
            W /= W.norm(dim=0, keepdim=True).clamp_min(1e-6)
        args._state = dict(step=0, W=W)

    st, W, step = args._state, args._state["W"], args._state["step"]

    # projector
    def P(vec):
        return W @ (W.T @ vec)

    # keep a sane base step (no rank/epsilon amplification)
    lr_eta   = lr_sched
    macro_bs = getattr(args, "macro_batch_size", 1)
    verbose  = bool(args and getattr(args, "verbose", False))
    start_t  = time.time()

    # ───── Baseline loss at θ (for safety/backtracking) ─────
    vector_to_parameters(model_params_flat, model_params)
    base_loss = 0.0
    finite0 = True
    for _ in range(macro_bs):
        lv = float(loss_fn())
        if not math.isfinite(lv):
            finite0 = False
            break
        base_loss += lv
    base_loss = base_loss / macro_bs if finite0 else float("inf")

    # ───── 1. SPSA gradient estimate ─────
    g_flat = torch.zeros_like(model_params_flat, dtype=torch.float32)
    seeds    = [random.randint(0, 2**32 - 1) for _ in range(num_perturbations)]
    safe_eps = max(1e-8, float(abs(epsilon)))

    for sd in seeds:
        # IMPORTANT: keep true Rademacher scale (±1 per coordinate)
        probe = generate_perturbation(model_params_flat, 1, distribution, sd).to(torch.float32)

        if hasattr(args, "use_probe_preconditioning") and args.use_probe_preconditioning:
            # project into subspace then rescale to match ||rad|| ≈ √d
            probe = P(probe)
            pn = probe.norm()
            if torch.isfinite(pn) and pn > 0:
                probe.mul_(math.sqrt(float(d)) / (pn + 1e-12))

        # +ε
        with torch.no_grad():
            model_params_flat.add_(probe, alpha=safe_eps)
            vector_to_parameters(model_params_flat, model_params)
        pos, finite = 0.0, True
        for _ in range(macro_bs):
            lv = float(loss_fn())
            if not math.isfinite(lv):
                finite = False
                break
            pos += lv
        pos = pos / macro_bs if finite else float("nan")

        # –ε
        with torch.no_grad():
            model_params_flat.add_(probe, alpha=-2.0 * safe_eps)
            vector_to_parameters(model_params_flat, model_params)
        neg = 0.0
        if finite:
            for _ in range(macro_bs):
                lv = float(loss_fn())
                if not math.isfinite(lv):
                    finite = False
                    break
                neg += lv
            neg = neg / macro_bs if finite else float("nan")

        # restore θ
        with torch.no_grad():
            model_params_flat.add_(probe, alpha=safe_eps)
            vector_to_parameters(model_params_flat, model_params)

        if not finite:
            continue

        # keep your sign: g_flat will approximate -∇f
        coeff = -(pos - neg) / (2.0 * num_perturbations * safe_eps)
        g_flat.add_(probe, alpha=coeff)

    g_norm = g_flat.norm()

    # optional clip
    # if clip_grad_norm and clip_grad_norm > 0.0:
    #     if torch.isfinite(g_norm) and g_norm > clip_grad_norm:
    #         g_flat.mul_(clip_grad_norm / (g_norm + 1e-12))
    #         g_norm = g_flat.norm()

    # ───── 2. pre-condition & update θ ─────
    pre_g  = P(g_flat) + alpha_eye_scalar * g_flat  

    # Scale update so L2(Δθ) ≈ ε·√d (≈ ε per parameter on average)
    update = pre_g * lr_eta
    # upn = update.norm()
    # target_norm = safe_eps * math.sqrt(float(d))
    # if torch.isfinite(upn) and upn > 0:
    #     update.mul_(target_norm / (upn + 1e-12))

    # Safety backtracking only if loss explodes; otherwise accept to avoid freezing
    # accepted = False
    # for _try in range(6):
    with torch.no_grad():
        model_params_flat.add_(update)
        vector_to_parameters(model_params_flat, model_params)

    #     new_loss, finite = 0.0, True
    #     for _ in range(macro_bs):
    #         lv = float(loss_fn())
    #         if not math.isfinite(lv):
    #             finite = False
    #             break
    #         new_loss += lv
    #     new_loss = new_loss / macro_bs if finite else float("inf")

    #     # If non-finite or catastrophically worse, backtrack; else accept
    #     if (not finite) or (new_loss > base_loss * 5.0):
    #         with torch.no_grad():
    #             model_params_flat.sub_(update)
    #             vector_to_parameters(model_params_flat, model_params)
    #         update.mul_(0.25)
    #     else:
    #         accepted = True
    #         base_loss = min(base_loss, new_loss)
    #         break

    # if not accepted:
    #     # last-ditch tiny step in the descent direction to keep moving
    #     tiny = (pre_g / pre_g.norm().clamp_min(1e-12)) * (safe_eps / math.sqrt(float(d)))
    #     with torch.no_grad():
    #         model_params_flat.add_(tiny)
    #         vector_to_parameters(model_params_flat, model_params)

    # ───── 3. memory-light W update (column-wise) ─────
    with torch.no_grad():
        g_unit = g_flat / g_norm.clamp_min(1e-12)
        proj   = W.T @ g_unit
        acc    = torch.zeros_like(g_unit)

        for i in range(rank):
            acc.add_(W[:, i], alpha=proj[i])    # Σ_{j≤i} proj_j W_j
            delta = (g_unit - acc) * proj[i]
            W[:, i].add_(delta, alpha=β_eig)

            # keep each column well-scaled
            cn = W[:, i].norm()
            if torch.isfinite(cn) and cn > 0:
                W[:, i].div_(cn.clamp_min(1e-6))

        # periodic QR (best-effort)
        if (step + 1) % args.sanger_qr_every == 0:
            try:
                W, _ = torch.linalg.qr(W, mode="reduced")
                
            except RuntimeError:
                pass

        # cheap variance metric (no QR)
        denom = g_flat.pow(2).sum().clamp_min(1e-12)
        var_ratio = ((W.T @ g_flat).pow(2).sum() / denom).clamp(min=0.0, max=1.0)
        if True:
            print(f"Subspace captures {(var_ratio * 100).item():.3f}% of gradient variance")

    st["W"], st["step"] = W, step + 1

    # ───── 4. logging / return ─────
    if verbose:
        dt = time.time() - start_t
        print(f"[Sanger-SPSA-W] step={step:4d} "
              f"loss={base_loss:.5f} "
              f"var%={(var_ratio*100).item():.2f}  Δt={dt:.2f}s")

    return dict(iterations=[step],
                train_loss=[float(base_loss)],
                elapsed_time=[time.time() - start_t],
                grad_norm=[float(g_norm.item()) if torch.isfinite(g_norm) else float('nan')])



def train_step(model, args, batch_np, loss_closure, optimizer=None):
    """
    One optimisation step, agnostic to LSTM/DNC and to Zeroth-/First-order.
    `loss_closure()` must compute loss **using the CURRENT parameters**.
    """
    if args.solver=="BPTT":                   # ---------- first-order -----------
        if optimizer is None:
            raise ValueError("optimizer must be provided when using sovler==bptt")

        for p in model.parameters():
            p.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.macro_batch_size):
            loss = loss_closure()
            (loss / args.macro_batch_size).backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        return {"train_loss": [total_loss], "grad_norm": [0.0]}

        
    elif args.solver=="1SPSA":
        # ------------------- zeroth-order (CD-RGE / FDRAS) --------------------
        with torch.no_grad():
            # can also replace this with cdrge_optimize, cdrge_no_chunking, or fdras_optimized depending on what you want to test
             
            return cdrge_optimize( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate,
                epsilon           = args.epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = True,
                args              = args,
            )
            
    elif args.solver=="1.5-SPSA":
        # ------------------- zeroth-order (CD-RGE / FDRAS) --------------------
        with torch.no_grad():
            return SPSA1_5( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate,
                epsilon           = args.epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = True,
                args              = args,
            )
            
    elif args.solver=="2SPSA":
        with torch.no_grad():
            return SPSA2( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate,
                epsilon           = args.epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = True,
                args              = args,
            )
            
    elif args.solver=="BanditSPSA":
        with torch.no_grad():
            return BanditSPSA( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate,
                epsilon           = args.epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = True,
                args              = args,
            ) 


    elif args.solver=="Sanger-SPSA":
        with torch.no_grad():
            return SangerSPSA( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate,
                epsilon           = args.epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = True,
                args              = args,
            ) 
    
    else:
        raise Exception(f"no solver implemented named {args.solver}")




def evaluate(model, tok, args, split="val", show_predictions=True, num_samples=3):
    with torch.no_grad():
        batch_np = get_examples_for_task(
            args.task, tok, args.micro_batch_size, args.seq_length,
            split=split, max_num=args.max_num
        )
        ids  = torch.as_tensor(batch_np, device=args.device, dtype=torch.long)
        xemb = model.embed(ids)
        logits, _, _ = model(xemb, require_gradients=False)

        loss     = compute_task_loss(logits, batch_np, tok, args.task)
        accuracy = compute_task_accuracy(logits, batch_np, tok, args.task)

        if show_predictions and args.task in ["copy","repeat_copy","sort","reverse","add"]:
            preds = torch.argmax(logits, dim=-1)

            if args.tokenizer == "hf_tiktoken":
                decode_fn = lambda ids: tok.decode(ids.tolist())
                decoded_preds = [decode_fn(seq) for seq in preds]
                decoded_inputs = [decode_fn(seq) for seq in ids]
                decoded_targets = [decode_fn(seq) for seq in torch.as_tensor(batch_np)]
            else:
                id_to_char = tok.id_to_char if hasattr(tok, "id_to_char") else tok.id_to_token
                decode_fn = lambda ids: "".join([id_to_char.get(i.item(), '') for i in ids])
                decoded_preds = [decode_fn(seq) for seq in preds]
                decoded_inputs = [decode_fn(seq) for seq in ids]
                decoded_targets = [decode_fn(seq) for seq in torch.as_tensor(batch_np)]

            print("=" * 40)
            print(f"Validation predictions ({split} split):")
            for i in range(min(len(decoded_preds), num_samples)):
                print(f"[Sample {i}] Input:     '{decoded_inputs[i]}'")
                print(f"[Sample {i}] Target:    '{decoded_targets[i]}'")
                print(f"[Sample {i}] Predicted: '{decoded_preds[i]}'")
            print("=" * 40)

        return {"loss": loss, "accuracy": accuracy}




def train(args):
    """Main training function"""
    # Set device
    device = torch.device(args.device)

    # Initialize wandb for experiment tracking (with defaults to avoid attribute errors)
    wandb_enabled = False

    # Check for existence of wandb attributes
    has_wandb_proj = hasattr(args, 'wandb_proj') and args.wandb_proj is not None
    has_api_key = hasattr(args, 'WANDB_API_KEY')
    has_run_name = hasattr(args, 'wandb_run_name')

    # Only enable wandb if all required attributes exist and conditions are met
    if WANDB_AVAILABLE and has_wandb_proj and args.wandb:
        # Set API key if available
        if has_api_key:
            os.environ["WANDB_API_KEY"] = args.WANDB_API_KEY

        # Determine run name
        if not has_run_name or args.wandb_run_name is None:
            run_name = f"{args.task}_{args.hidden_size}_{args.distribution}_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            run_name = args.wandb_run_name

        
        # Initialize wandb
        wandb.init(
            project=args.wandb_proj,
            name=run_name,
            config=vars(args),
            reinit=True
        )
        wandb_enabled = True
        
        print(f"Weights & Biases logging enabled. Project: {args.wandb_proj}, Run: {run_name}")
    else:
        # Don't show warnings for missing attributes in test mode
        print("No Weights & Biases logging enabled.")
    
    # Initialize tokenizer
    if args.tokenizer == "char_level" or not TIKTOKEN_AVAILABLE:
        if not TIKTOKEN_AVAILABLE and args.tokenizer == "hf_tiktoken":
            print("Warning: tiktoken is not available. Falling back to char_level tokenizer.")

        # Use CharTokenizer for sort/copy/reverse (letter-based tasks)
        # And NumericTokenizer only for add (number-based task)
        if args.task == "add":
            tok = NumericTokenizer(max_num=args.max_num)
        else:
            tok = CharTokenizer()
    else:  # hf_tiktoken
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            class TiktokenWrapper:
                def __init__(self, enc):
                    self.enc = enc
                    self.vocab_size = enc.n_vocab

                def encode(self, text):
                    return self.enc.encode(text)

                def decode(self, tokens):
                    return self.enc.decode(tokens)

            tok = TiktokenWrapper(enc)
        except Exception as e:
            print(f"Failed to load tiktoken: {str(e)}. Falling back to char_level tokenizer.")
            if args.task in ["add", "sort", "reverse"]:
                tok = NumericTokenizer(max_num=args.max_num)
            else:
                tok = CharTokenizer()
    
    dtype = torch.float32 if args.solver=="BPTT" else torch.float32 # torch.float16 # bptt in fp16 is unstable... try it but be weary.. vanishing/exploding grads await you 

    # ───────── Model construction (LSTM or DNC) ─────────
    embed = nn.Embedding(tok.vocab_size, args.input_size,
                         device=device, dtype=dtype)
    
    if args.model_type.upper() == "LSTM":
        model = LSTM(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    elif args.model_type.upper() == "DNC":
        model = DNC(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    
    elif args.model_type == "Transformer":
        model = Transformer(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    elif args.model_type == "Mamba":
        model = Mamba(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    elif args.model_type == "SSM":
        model = SSM(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    else:
        raise ValueError(f"Unknown model_type {args.model_type}")
    
    model_params = list(model.parameters())          # <- used by zeroth-order
    param_count  = sum(p.numel() for p in model_params)
    bytes_per_param = 4 if args.solver=="BPTT" else 2
    param_memory_mb = param_count * bytes_per_param / (1024*1024)
    
    print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

    
    
    
    # Determine bytes per parameter based on optimizer type
    if args.solver=="BPTT":
        bytes_per_param = 4  # float32 = 4 bytes
    else:
        bytes_per_param = 4 
        # bytes_per_param = 2  # bfloat16 = 2 bytes
        
    
    # Set up results dictionary
    results = {
        'args': vars(args),
        'train_metrics': {
            'loss': [],
            'accuracy': [],
            'iterations': [],
            'time': []
        },
        'val_metrics': {
            'loss': [],
            'accuracy': [],
            'iterations': []
        }
    }
    
    # Main training loop
    print(f"Starting training for task: {args.task}")
    print("="*50)
    print(args)
    print("="*50)

    total_start_time = time.time()

    has_overfit_flag = hasattr(args, 'overfit_to_one_batch_flag') and args.overfit_to_one_batch_flag

    # Generate first batch to compute initial loss
    
    init_batch = get_examples_for_task(
        args.task, tok, args.micro_batch_size, args.seq_length,
        split='train', max_num=args.max_num
    )
    
    if has_overfit_flag:
        print("="*50)
        print("OVERFITTING TO A SINGLE BATCH!")
        

        train_batch = init_batch 
        # sep_token = tokenizer.token_to_id.get(" ", 0) 

    # ───────── Initial loss / accuracy BEFORE training ─────────
    with torch.no_grad():
        ids_init   = torch.as_tensor(init_batch, device=device, dtype=torch.long)
        x_emb_init = model.embed(ids_init)               # shared embed layer
        init_logits, _, _ = model(x_emb_init, require_gradients=False)            # works for LSTM & DNC
    
        initial_loss     = compute_task_loss(init_logits, init_batch, tok, args.task)
        initial_accuracy = compute_task_accuracy(init_logits, init_batch, tok, args.task)


    print(f"Initial loss (before training): {initial_loss:.4f}, Accuracy: {initial_accuracy:.4f}")

    # Log initial metrics to wandb
    if wandb_enabled:
        wandb.log({
            "initial_loss": initial_loss,
            "initial_accuracy": initial_accuracy
        }, step=0)

    # Store initial metrics in results
    results['train_metrics']['loss'].append(initial_loss)
    results['train_metrics']['accuracy'].append(initial_accuracy)
    results['train_metrics']['iterations'].append(0)
    results['train_metrics']['time'].append(0)
    status = "training" 

    require_gradients = False
    if args.solver == "BPTT":
        require_gradients = True
        if args.use_adam:
            # Set up optimizer with memory-efficient settings
            # Use AdamW with lower memory footprint than standard Adam
            print("Using AdamW")
            optimizer = torch.optim.AdamW(
                model_params, 
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                eps=1e-8,  # Slightly larger epsilon for stability
                weight_decay=args.weight_decay,  
                amsgrad=False  # Disable amsgrad to save memory
            )
        else:
            print("Using vanilla SGD")
            optimizer = torch.optim.SGD(
                model_params, 
                lr=args.learning_rate,
                momentum=args.beta1, 
                weight_decay=args.weight_decay
            )
    else:
        optimizer = None

    if args.hidden_size<4096:
        args.cache_gradients = True # THIS WILL COST US VRAM, but speed us up at the end, if we have the space we should use it.
    else:
        args.cache_gradients = False # THIS WILL SAVE VRAM, but slow us down a bit at the end.

    # ───────── Binary LR Search Initialization ─────────
    plateau_detector = None
    if hasattr(args, 'binary_lr_search') and args.binary_lr_search and args.solver in ["1SPSA", "1.5-SPSA"]:
        print("="*50)
        print("BINARY LR SEARCH MODE ENABLED")
        print("="*50)

        # Create plateau detector
        plateau_detector = EMAPlateauDetector(
            alpha=args.plateau_ema_alpha if hasattr(args, 'plateau_ema_alpha') else 0.1,
            patience=args.plateau_patience if hasattr(args, 'plateau_patience') else 10,
            threshold=args.plateau_threshold if hasattr(args, 'plateau_threshold') else 0.01
        )

        # Define loss function for LR search (using init_batch for consistency)
        def lr_search_loss_fn():
            ids = torch.as_tensor(init_batch, device=device, dtype=torch.long)
            xemb = model.embed(ids)
            logits, _, _ = model(xemb, require_gradients=False)
            return compute_task_loss(logits, init_batch, tok, args.task)

        # Run initial binary LR search
        print("Running initial binary LR search...")
        best_lr = binary_search_lr(
            model_params=model_params,
            loss_fn=lr_search_loss_fn,
            lr_min=args.lr_search_min if hasattr(args, 'lr_search_min') else 1e-5,
            lr_max=args.lr_search_max if hasattr(args, 'lr_search_max') else 0.1,
            num_perturbations=args.num_perturbations,
            distribution=args.distribution,
            args=args,
            depth=args.lr_search_depth if hasattr(args, 'lr_search_depth') else 3,
            seed=args.seed,
            probe_steps=args.lr_search_probe_steps if hasattr(args, 'lr_search_probe_steps') else 1,
            cache_weights_to_cpu=args.cache_weights_to_cpu if hasattr(args, 'cache_weights_to_cpu') else False
        )

        # Update LR and epsilon
        args.learning_rate = best_lr
        args.epsilon = best_lr
        print(f"Initial LR search complete. Using lr={best_lr:.2e}")
        print("="*50)

    total_iterations = 0
    for iteration in range(args.max_iterations):
        # Get batch based on task (or reuse same batch if overfitting)
        start_time = time.time()
        if not has_overfit_flag:
            train_batch = get_examples_for_task(
                args.task, tok, args.micro_batch_size, args.seq_length,
                split='train', max_num=args.max_num
            )
                
        # Define loss function for current batch
        def compute_loss(return_acc=False):
            ids  = torch.as_tensor(train_batch, device=device, dtype=torch.long)
            xemb = model.embed(ids)                     # [B,T,E]
            logits, _, _ = model(xemb, require_gradients=require_gradients)                  # same for LSTM & DNC
        
            loss = compute_task_loss(logits, train_batch, tok, args.task)
            if return_acc:
                acc = compute_task_accuracy(logits, train_batch, tok, args.task)
                return loss, acc
            return loss
    
        
        step_metrics = train_step(model, args, train_batch, compute_loss, optimizer)
        train_accuracy = -1.0  # or calculate here if you want but loss is fine for me
        
        # Store metrics
        results['train_metrics']['loss'].append(step_metrics['train_loss'][-1])
        results['train_metrics']['accuracy'].append(train_accuracy) # placeholder for now... 
        results['train_metrics']['iterations'].append(iteration + 1)
        results['train_metrics']['time'].append(time.time() - start_time)
        
        # Calculate iteration time and other metrics
        iteration_time = time.time() - start_time
        total_elapsed = time.time() - total_start_time


        # Get actual GPU memory usage if available
        max_gpu_memory_mb = 0
        if torch.cuda.is_available():
            max_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            # print(f"Max GPU memory used: {max_gpu_memory_mb:.1f} MB")

        print(f"Iteration {iteration+1}/{args.max_iterations} | "
              f"Train Loss: {step_metrics['train_loss'][-1]:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | "
              f"iteration time: {iteration_time:.4f}")

        print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

        if True: #has_overfit_flag:
            if step_metrics['train_loss'][-1] < 0.1:
                print(f"SUCCESS FINISHED! Overfit in iterations = {iteration}")
                print(args)
                total_iterations = iteration
                status = "success"
                
                # time.sleep(100000)
                # return
                break
            elif step_metrics['train_loss'][-1] > 7. or math.isnan(step_metrics['train_loss'][-1]):
                print(f"FAILED DIVERGING!")
                print(args)
                total_iterations = iteration
                status = "diverged"
                # time.sleep(100000)
                # return
                break

        # ───────── Plateau Detection and LR Re-search ─────────
        if plateau_detector is not None:
            current_loss = step_metrics['train_loss'][-1]
            is_plateau = plateau_detector.update(current_loss)

            if is_plateau:
                print("="*50)
                print(f"PLATEAU DETECTED at iteration {iteration+1} (EMA: {plateau_detector.ema:.4f})")
                print("Running binary LR search...")

                # Define loss function for LR search
                def lr_search_loss_fn_plateau():
                    ids = torch.as_tensor(train_batch, device=device, dtype=torch.long)
                    xemb = model.embed(ids)
                    logits, _, _ = model(xemb, require_gradients=False)
                    return compute_task_loss(logits, train_batch, tok, args.task)

                # Run binary LR search
                best_lr = binary_search_lr(
                    model_params=model_params,
                    loss_fn=lr_search_loss_fn_plateau,
                    lr_min=args.lr_search_min if hasattr(args, 'lr_search_min') else 1e-5,
                    lr_max=args.lr_search_max if hasattr(args, 'lr_search_max') else 0.1,
                    num_perturbations=args.num_perturbations,
                    distribution=args.distribution,
                    args=args,
                    depth=args.lr_search_depth if hasattr(args, 'lr_search_depth') else 3,
                    seed=args.seed + iteration,  # Different seed each time
                    probe_steps=args.lr_search_probe_steps if hasattr(args, 'lr_search_probe_steps') else 1,
                    cache_weights_to_cpu=args.cache_weights_to_cpu if hasattr(args, 'cache_weights_to_cpu') else False
                )

                # Update LR and epsilon
                old_lr = args.learning_rate
                args.learning_rate = best_lr
                args.epsilon = best_lr
                print(f"LR updated: {old_lr:.2e} -> {best_lr:.2e}")
                print("="*50)

                # Reset plateau detector
                plateau_detector.reset()

        # Logging
        if (iteration + 1) % args.log_interval == 0 and not has_overfit_flag:
            with torch.no_grad():
                # Validation (every 10% of iterations or every 10 iterations for short runs)
                # eval_interval = max(1, min(10, args.max_iterations // 10))
                # During training, don't show predictions to save space
                val_metrics = evaluate(model, tok, args, split='val', show_predictions=True)


                # val_metrics = evaluate(model_params, tok, args, split='val', show_predictions=True)
                results['val_metrics']['loss'].append(val_metrics['loss'])
                results['val_metrics']['accuracy'].append(val_metrics['accuracy'])
                results['val_metrics']['iterations'].append(iteration + 1)

            print(f"Validation at iter {iteration+1}: "
                  f"Loss: {val_metrics['loss']:.4f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")

            print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

            # Log metrics to wandb
            if wandb_enabled:
                # Basic metrics
                wandb_metrics = {
                    "train_loss": step_metrics['train_loss'][-1],
                    "train_acc": train_accuracy,
                    "lr": args.learning_rate,
                    "epsilon": args.epsilon,
                    "iter_time_s": iteration_time,
                    "total_time_hours": total_elapsed / 3600.0,
                    "grad_norm": step_metrics.get('grad_norm', [0])[-1] if step_metrics.get('grad_norm') else 0,
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                }
                wandb.log(wandb_metrics, step=iteration + 1)



    # final cleanup after training
    with torch.no_grad():
        ids_final   = torch.as_tensor(init_batch, device=device, dtype=torch.long)
        x_emb_final = model.embed(ids_final)
        final_logits, _, _ = model(x_emb_final, require_gradients=False)
        final_loss     = compute_task_loss(final_logits, init_batch, tok, args.task)
        final_accuracy = compute_task_accuracy(final_logits, init_batch, tok, args.task)
        
    results['train_metrics']['loss'].append(final_loss)
    results['train_metrics']['accuracy'].append(final_accuracy)
    results['train_metrics']['iterations'].append(total_iterations)
    results['train_metrics']['time'].append(0)
    results['param_count'] = param_count
    
    results["status"] = status

    print(f"Final loss (after training): {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    
    # Log final results
    total_time = time.time() - total_start_time
    print("\nTraining completed!")
    print(f"Loss reduction: {results['train_metrics']['loss'][0] - results['train_metrics']['loss'][-1]:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")

    print(" DONE! Now go get the actual vram")
    time.sleep(10)
    # --- drop strong Python refs --------------------------------------------
    if 'model_params' in locals():
        del model_params           # list of tensors
    if 'optimizer' in locals() and optimizer is not None:
        del optimizer
    
    # --- force Python to release any lingering tensors -----------------------
    import gc
    gc.collect()                   # releases refs held only by GC
    
    # --- tell the CUDA caching allocator to give memory back to the pool ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # frees cached blocks
        torch.cuda.ipc_collect()   # optional: reclaims inter-process buffers

    # print("="*50)
    # print("all results")
    # print(results)
    # print("="*50)
    
    return results
    

import argparse, time, math, traceback
import torch
import argparse, time, math, torch


def run_unittest(args):
    """Run a battery of quick sanity tests on `train()`."""
    
    batch_size = args.micro_batch_size
     

    # ---------------------------------------------------------------------- #
    #  All experiment configurations                                         #
    # ---------------------------------------------------------------------- #
    all_runs = {

            ### DNC TESTS
            # ========================= DNC BPTT ===================================== #
            # "tiny-bptt-default-dnc":   dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
            #                             solver="BPTT", use_adam=False, learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Tiny dnc with BPTT"),
            # "small-bptt-default-dnc":  dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, model_type="DNC",
            #                            solver="BPTT", use_adam=False, learning_rate=0.01, epsilon=0.01, learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Small dnc with BPTT "),
            # "medium-bptt-default-dnc": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, model_type="DNC",
            #                             solver="BPTT", use_adam=False, learning_rate=0.0001, epsilon=0.0001,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Medium dnc with BPTT "),
            # "medium-bptt-default-dnc": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, model_type="DNC",
            #                                 solver="BPTT", use_adam=True, learning_rate=0.0001, epsilon=0.0001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                 description="Medium dnc with BPTT with Adam"),
            
            # ========================= DNC cdrge-96 ================================= #
            # "tiny-cdrge96-default-dnc":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
            #                               learning_rate=0.01, epsilon=0.01,   #.01 for 1SPSA,  
            #                                   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Tiny dnc with cdrge@96"),
            # "small-cdrge96-default-dnc": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, memory_size=128, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Small dnc with cdrge@96"),
            # "medium-cdrge96-default-dnc":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, memory_size=128, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Medium dnc with cdrge@96"),
            # "large-cdrge96-default-dnc": dict(model_size="large",  hidden_size=2**13, num_heads=1,  head_size=0, memory_size=2**13, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Large dnc with cdrge@96"),
            # 2.4B  with  hidden_size=2**14, num_heads=2**8, head_size=0,
            # "xlarge-cdrge96-default-dnc":dict(model_size="xlarge", hidden_size=2**14, num_heads=2**8, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XLarge dnc with cdrge@96"),
            # 10B  with  hidden_size=2**15, num_heads=2**9, head_size=0,
            # "xxlarge-cdrge96-default-dnc":dict(model_size="xxlarge", hidden_size=2**15, num_heads=2**10, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate=0.0001, epsilon=0.0001, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XXLarge dnc with cdrge@96"),
        
            # ========================= DNC cdrge-512 ================================ #
            # "tiny-cdrge512-default-dnc":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
            #                                learning_rate=0.01, epsilon=0.01, #.1 for 1SPSA,  
            #                                    micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Tiny dnc with cdrge@512"),
            # "small-cdrge512-default-dnc": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, memory_size=128, model_type="DNC",
            #                                learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Small dnc with cdrge@512"),
            # "medium-cdrge512-default-dnc":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, memory_size=128, model_type="DNC",
            #                                learning_rate=0.001, epsilon=0.001, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Medium dnc with cdrge@512"),
            # "large-cdrge512-default-dnc": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300, memory_size=128, model_type="DNC",
            #                                learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Large dnc with cdrge@512"),
            # 2.4B  with  hidden_size=2**14, num_heads=2**8, head_size=0,
            # "xlarge-cdrge512-default-dnc":dict(model_size="xlarge", hidden_size=2**14, num_heads=2**8, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=512,  antithetic=False,
            #                               description="XLarge dnc with cdrge@96"),
            # 10B  with  hidden_size=2**15, num_heads=2**9, head_size=0,
            # "xxlarge-cdrge96-default-dnc":dict(model_size="xxlarge", hidden_size=2**15, num_heads=2**10, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate=0.001, epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=512,  antithetic=False,
            #                               description="XXLarge dnc with cdrge@512"),

            ### LSTM TESTS
            # ========================= LSTM BPTT ===================================== #
            
            # "tiny-bptt-default-lstm":   dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, model_type="LSTM",
            #                             solver="BPTT", use_adam=True, learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1, 
            #                             description="Tiny lstm with BPTT"),
            # "small-bptt-default-lstm":  dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, model_type="LSTM",
            #                             solver="BPTT", use_adam=True, learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Small lstm with BPTT"),
            # "medium-bptt-default-lstm": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                             solver="BPTT", use_adam=True, learning_rate=0.001, epsilon=0.001, micro_batch_size=int(batch_size/128), macro_batch_size=128,
            #                             description="Medium lstm with BPTT - optimized for GPU memory constraints"),
            
            # ========================= LSTM cdrge-96 ================================= #
            "tiny-banditspsa-96-default-lstm":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, model_type="LSTM",
                                          learning_rate=0.001, epsilon=0.001,    micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                          num_perturbations=96,  antithetic=False, seed=42, solver="BanditSPSA", beta1=0.0, beta2=0.0, 
                                          use_probe_preconditioning=False, description="Tiny lstm with cdrge@96"),
            # "tiny-cdrge96-default-lstm":  dict(model_size="tiny",   hidden_size=2400,   num_heads=12,  head_size=20,model_type="LSTM",
            #                               learning_rate=0.1, epsilon=0.1,    micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False, seed=42, solver="1SPSA", beta1=0.0, beta2=0.0, 
            #                               use_probe_preconditioning=False, description="Tiny lstm with cdrge@96"),

        
            # "tiny-cdrge96-default-lstm-with-rmsprop":  dict(model_size="tiny",   hidden_size=2400,   num_heads=12,  head_size=20,model_type="LSTM",
            #                               learning_rate=0.1, epsilon=0.1,    micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False, seed=42, solver="1SPSA", beta1=0.0, beta2=0.1, 
            #                               use_probe_preconditioning=False, description="Tiny lstm with cdrge@96"),
            # "tiny-cdrge96-default-lstm-with-rmsprop-probe-precond":  dict(model_size="tiny",   hidden_size=2400,   num_heads=12,  head_size=20,model_type="LSTM",
            #                               learning_rate=0.1, epsilon=0.1,    micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False, seed=42, solver="1SPSA", beta1=0.0, beta2=0.1, 
            #                               use_probe_preconditioning=True, description="Tiny lstm with cdrge@96"),
            # "small-cdrge96-default-lstm": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50,model_type="LSTM",
            #                               learning_rate=0.01, epsilon=0.01, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Small lstm with cdrge@96"),
            # "medium-cdrge96-default-lstm":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                               learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Medium lstm with cdrge@96"),
            # "large-cdrge96-default-lstm": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300,model_type="LSTM",
            #                               learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Large lstm with cdrge@96"),
            # "xlarge-cdrge96-default-lstm":dict(model_size="xlarge", hidden_size=297500, num_heads=350, head_size=850,model_type="LSTM",
            #                               learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XLarge lstm with cdrge@96"),
        
            # ========================= LSTM cdrge-512 ================================ #
            # "tiny-cdrge512-default-lstm":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20,model_type="LSTM",
            #                                learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False, solver="1SPSA", 
            #                                description="Tiny lstm with cdrge@512"),
            # "small-cdrge512-default-lstm": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50,model_type="LSTM",
            #                                learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Small lstm with cdrge@512"),
            # "medium-cdrge512-default-lstm":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                                learning_rate=0.01, epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Medium lstm with cdrge@512"),
            # "large-cdrge512-default-lstm": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300,model_type="LSTM",
            #                                learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Large lstm with cdrge@512"),
            # "xlarge-cdrge512-default-lstm":dict(model_size="xlarge", hidden_size=297500, num_heads=350, head_size=850,model_type="LSTM",
            #                                learning_rate=0.01, epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="XLarge lstm with cdrge@512"),


    }
    


    # ---------------------------------------------------------------------- #
    #  Main loop                                                             #
    # ---------------------------------------------------------------------- #
    failures = []
    for run_name, run_cfg in all_runs.items():
        print("="*50)
        print("="*50)
        print("="*50)
        print(f"\n=== {run_name}: {run_cfg['description']} ===")
    
        # Get a fresh copy of default CLI args
        run_args = vars(get_argument_parser())

        run_args["overfit_to_one_batch_flag"] = True
        
        # Override defaults with specific run config
        run_args.update(run_cfg)
    
        # Create final args object
        args = argparse.Namespace(**run_args)
        print(args)

        try:
            t0 = time.time()
            results = train(args)                     
            total_time = time.time() - t0
            avg_time   = total_time / args.max_iterations

            # ---------------- aggregate metrics -------------------------- #
            train_losses = results['train_metrics']['loss']
            val_losses   = results['val_metrics']['loss']

            train_delta = train_losses[0] - train_losses[-1]
            val_delta   = (val_losses[0] - val_losses[-1]) if len(val_losses) >= 2 else float('nan')

            print(f"Total time: {total_time:.2f}s | Avg/iter: {avg_time:.4f}s | "
                  # f"≈VRAM: {est_vram_gb:.2f} GB | "
                  f"ΔTrain-loss: {train_delta:+.4f} | ΔVal-loss: {val_delta:+.4f}")

            # ---------------- basic assertions --------------------------- #
            if train_delta <= 0:
                raise AssertionError("training loss did not improve")
            if not math.isnan(val_delta) and val_delta <= 0:
                raise AssertionError("validation loss did not improve")


            print(f"{run_name} ✅ passed!")
        except RuntimeError as e:
            print("="*50)
            print("="*50)
            print("="*50)
            failures.append((run_name, str(e)))
            print(f"❌  {run_name} FAILED: {e}")
            print("="*50)
            print("="*50)
            print("="*50)
            traceback.print_exc()
            # PyTorch 2.1+ raises a dedicated subclass
            oom = isinstance(e, torch.cuda.OutOfMemoryError) \
                  or "out of memory" in str(e).lower()
        
            if oom:
                raise RuntimeError(
                    "\nCUDA OOM detected.\n"
                    "  • First, try setting `args.cache_gradients = True` "
                    "or lowering `CHUNK_SIZE`.\n"
                    "  • If that’s not enough, reduce `micro_batch_size` and raise "
                    "`macro_batch_size` to keep the same total tokens.\n"
                    "  • As a last resort, choose a smaller model (hidden_size / heads).\n"
                ) from e
            else:
                # not an OOM → re-raise unchanged
                raise
            
            # return
            # Continue running the remaining tests instead of aborting immediately

    # ---------------------------------------------------------------------- #
    #  Summary                                                               #
    # ---------------------------------------------------------------------- #
    if failures:
        print("\n================ FAILED TESTS ================")
        for name, err in failures:
            print(f"{name}: {err}")
        raise RuntimeError(f"{len(failures)} test(s) failed.")
    else:
        print("\n✅  All unit tests passed!")





# ───────── Main ─────────
def get_argument_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(description="Train neural models on basic functions")

    # Task and model configuration
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="penn_tree_bank",
                        choices=["copy", "repeat_copy", "sort", "reverse", "add", "penn_tree_bank"])
    parser.add_argument("--model_type", type=str, default="LSTM",
                        choices=["LSTM", "DNC", "Transformer", "Mamba", "SSM"])
    parser.add_argument("--seq_length", type=int, default=10,
                        help="For generation.")
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--max_num", type=int, default=9,
                        help="This is the max number in the domain to use in training for arithmetic tasks.")

    # Model architecture
    parser.add_argument("--input_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=128)

    # Optimization settings
    parser.add_argument("--solver", type=str, default="1SPSA",
                        choices=["BPTT", "1SPSA", "1.5-SPSA", "2SPSA", "BanditSPSA", "Sanger-SPSA"] )
    parser.add_argument("--distribution", type=str, default="rad",
                        choices=["rad", "normal", "uniform"])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--saturating_alpha", type=float, default=1.0)
    parser.add_argument("--num_perturbations", type=int, default=20)
    parser.add_argument("--antithetic_sampling", action="store_true")
    parser.add_argument("--use_adaptive_step", action="store_true",
                        help="Use adaptive step sizes for FDRAS optimization") 
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.0, help="for momentum")
    parser.add_argument("--beta2", type=float, default=0.0, help="for RMSProp-style variance")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 weight decay")
    parser.add_argument("--use_probe_preconditioning", action="store_true", default=False,
                        help="Use the RMSProp g^2 EMA to precondition the probe.")
    parser.add_argument("--use_adam", action="store_true", default=False,
                        help="Use Adam optimizer vs. vanilla SGD")
    parser.add_argument("--overfit_to_one_batch_flag", action="store_true", default=False,
                        help="Use the same batch for all training iterations")
    parser.add_argument("--bandit_softmax_temperature", type=float, default=0.0001)
    parser.add_argument("--sanger_rank", type=int, default=1)
    parser.add_argument("--beta_eigen_sanger", type=float, default=0.1, help="for momentum")
    parser.add_argument("--sanger_qr_every", type=int, default=10000000)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--alpha_eye_scalar", type=float, default=1.)

    # Binary LR Search settings
    parser.add_argument("--binary_lr_search", action="store_true", default=False,
                        help="Enable adaptive LR search: find optimal LR at start and on plateau")
    parser.add_argument("--lr_search_min", type=float, default=1e-5,
                        help="Minimum LR to search (binary_lr_search)")
    parser.add_argument("--lr_search_max", type=float, default=0.1,
                        help="Maximum LR to search (binary_lr_search)")
    parser.add_argument("--lr_search_depth", type=int, default=3,
                        help="Depth of binary search (more depth = finer search)")
    parser.add_argument("--lr_search_probe_steps", type=int, default=1,
                        help="Number of steps to probe at each LR candidate")
    parser.add_argument("--plateau_patience", type=int, default=10,
                        help="Iterations without improvement before re-searching LR")
    parser.add_argument("--plateau_threshold", type=float, default=0.01,
                        help="Minimum relative improvement to consider progress")
    parser.add_argument("--plateau_ema_alpha", type=float, default=0.1,
                        help="EMA smoothing factor for plateau detection")
    parser.add_argument("--cache_weights_to_cpu", action="store_true", default=False,
                        help="Save weights to CPU during LR probing (saves GPU memory)")


    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_steps", type=int, default=5) 


    # Experiment tracking
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_proj", type=str, default="neural-functions")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", help="Print detailed debugging information")

    # Tokenization
    parser.add_argument("--tokenizer", type=str, default="char_level",
                       choices=["char_level", "numeric", "hf_tiktoken"])
    # Unit Test
    parser.add_argument("--unit_test", action="store_true")

    # Logging 
    parser.add_argument("--output_dir", type=str, default="./results_default/")
    parser.add_argument("--oom_backoff_sec", type=int, default=600) # 10 min
    parser.add_argument("--oom_max_retries", type=int, default=1000)


     # Parse the arguments
    args = parser.parse_args()
    
    return args


# OLD, SIMPLE VERSION OF MAIN
# def main():
#     """Parse arguments and run training"""
#     args = get_argument_parser()

#     # Set random seed
#     if args.seed == -1:
#         args.seed = np.random.randint(1, 1001) 
#         print(f"Setting seed to {args.seed}")

#     set_seed(args.seed)

#     if args.unit_test:
#         success = run_unittest(args)
#     else:
#         # Run training
#         results = train(args)

#         return results


import argparse, hashlib          
import json, time, traceback, pathlib, torch, numpy as np
import argparse, hashlib, json, time, traceback, pathlib
import torch, numpy as np

# ───────── Main ─────────
def main() -> None:
    """Train with OOM-retry loop and write results + full args to JSON."""
    args = get_argument_parser()

    # seed --------------------------------------------------------------
    if args.seed == -1:
        args.seed = np.random.randint(1, 1001)
        print(f"[INFO] Setting seed to {args.seed}")
    set_seed(args.seed)

    # unit test ---------------------------------------------------------
    if args.unit_test:
        run_unittest(args)
        return

    # output dir --------------------------------------------------------
    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    seed_hash = hashlib.md5((str(args.seed) + str(args)).encode()).hexdigest()[:6]
    run_prefix = args.wandb_run_name or "run"
    base_name = f"{run_prefix}_{seed_hash}"          # everything else added later

    backoff_sec = args.oom_backoff_sec
    max_retries = args.oom_max_retries
    attempt = 0

    while True:
        try:
            results = train(args)

            status = results["status"]
            iters  = int(results["train_metrics"]["iterations"][-1])
            filename = f"{status}_{base_name}_{iters}.json"
            if hasattr(args, "coordinate_momentum"):
                args.coordinate_momentum = None
            if hasattr(args, "coordinate_variance"):
                args.coordinate_variance = None
            if hasattr(args, "_state"):
                # print(args._state)
                args._state = None
                
            
            
            payload = {
                "status": status,
                "seed": int(args.seed),
                "iters": iters,
                "final_loss": float(results["train_metrics"]["loss"][-1]),
                "final_acc": float(results["train_metrics"]
                                   .get("accuracy", [0.0])[-1]),
                "wall_time_sec": float(results["train_metrics"]["time"][-1]),
                "wandb_run_name": run_prefix,
                "args": vars(args),
            }
            
            json.dump(payload, open(out_root / filename, "w"), indent=2)
            print(f"[INFO] Success → {filename} status {status} iters {iters}")
            return

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            attempt += 1
            if attempt > max_retries:
                filename = f"{base_name}_oom.json"
                json.dump({"status": "fail", "error": "OOM loop",
                           "args": vars(args)}, open(out_root / filename, "w"), indent=2)
                print(f"[FATAL] OOM loop → {filename}")
                raise
            traceback.print_exc()
            print(f"[OOM] attempt {attempt} — sleeping {backoff_sec//60} min")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(backoff_sec)

        except Exception as e:
            filename = f"{base_name}_fail.json"
            json.dump({"status": "fail", "error": str(e),
                       "args": vars(args)}, open(out_root / filename, "w"), indent=2)
            print(f"[ERROR] Exception → {filename}")
            raise


if __name__ == "__main__":
    main()

