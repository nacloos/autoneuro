"""Train a multi-task RNN on all extended Yang19 tasks."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from extended_yang19 import TASKS
import warnings
import sys
import os
import pickle
from model import MultiTaskRNN

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRAIN_TRIALS = 200   # per task
NUM_TEST_TRIALS = 50     # per task
HIDDEN_SIZE = 512
LR = 1e-3
EPOCHS = 60
BATCH_SIZE = 64
SEED = 42
DATA_CACHE_DIR = ".data_cache"

# ── Data generation ─────────────────────────────────────────────────────────
def generate_trials(task_name, task_fn, n_trials, max_steps=100):
    """Generate n_trials from a task, return (obs, gt, mask) arrays."""
    env = task_fn()
    env.reset(seed=None)
    
    all_obs, all_gt = [], []
    trial_obs, trial_gt = [], []
    trial_count = 0
    
    step = 0
    while trial_count < n_trials:
        ob, rew, term, trunc, info = env.step(0)
        trial_obs.append(ob)
        trial_gt.append(info.get('gt', 0))
        step += 1
        
        if info.get('new_trial', False) or step >= max_steps:
            if len(trial_obs) > 0:
                all_obs.append(np.array(trial_obs))
                all_gt.append(np.array(trial_gt))
                trial_count += 1
            trial_obs, trial_gt = [], []
            step = 0
    
    env.close()
    return all_obs, all_gt


def build_dataset(n_trials, seed=None, cache_name=None):
    """Build dataset with optional disk caching."""
    if cache_name:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(DATA_CACHE_DIR, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached {cache_name}...", file=sys.stderr)
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    
    if seed is not None:
        np.random.seed(seed)
    
    task_names = sorted(TASKS.keys())
    
    all_data = []
    for task_id, task_name in enumerate(task_names):
        task_fn = TASKS[task_name]
        obs_list, gt_list = generate_trials(task_name, task_fn, n_trials)
        for obs, gt in zip(obs_list, gt_list):
            all_data.append((obs, gt, task_id))
    
    result = (all_data, task_names)
    
    if cache_name:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Cached {cache_name} ({len(all_data)} sequences)", file=sys.stderr)
    
    return result


def collate_batch(batch, n_tasks):
    """Pad sequences to same length and create tensors."""
    obs_list, gt_list, task_ids = zip(*batch)
    
    max_len = max(o.shape[0] for o in obs_list)
    obs_dim = obs_list[0].shape[1]
    
    obs_padded = np.zeros((len(batch), max_len, obs_dim), dtype=np.float32)
    gt_padded = np.zeros((len(batch), max_len), dtype=np.int64)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    rule_input = np.zeros((len(batch), n_tasks), dtype=np.float32)
    
    for i, (obs, gt, tid) in enumerate(zip(obs_list, gt_list, task_ids)):
        T = obs.shape[0]
        obs_padded[i, :T] = obs
        gt_padded[i, :T] = gt
        mask[i, :T] = 1.0
        rule_input[i, tid] = 1.0
    
    return (
        torch.tensor(obs_padded, device=DEVICE),
        torch.tensor(gt_padded, device=DEVICE),
        torch.tensor(mask, device=DEVICE),
        torch.tensor(rule_input, device=DEVICE),
    )


# ── Training ────────────────────────────────────────────────────────────────
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("Generating training data...", file=sys.stderr)
    train_data, task_names = build_dataset(NUM_TRAIN_TRIALS, seed=SEED, cache_name=f"train_{NUM_TRAIN_TRIALS}_s{SEED}")
    n_tasks = len(task_names)
    
    print("Generating test data...", file=sys.stderr)
    test_data, _ = build_dataset(NUM_TEST_TRIALS, seed=SEED + 1000, cache_name=f"test_{NUM_TEST_TRIALS}_s{SEED+1000}")
    
    model = MultiTaskRNN(obs_dim=33, n_tasks=n_tasks, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)
    print(f"Training: {len(train_data)} sequences, Testing: {len(test_data)} sequences", file=sys.stderr)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(train_data)
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            obs, gt, mask, rule = collate_batch(batch, n_tasks)
            
            logits = model(obs, rule)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            gt_flat = gt.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, gt_flat, reduction='none')
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            train_acc = evaluate(model, train_data, n_tasks, task_names)
            test_acc = evaluate(model, test_data, n_tasks, task_names)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}", file=sys.stderr)
    
    # Final evaluation
    train_acc_final = evaluate(model, train_data, n_tasks, task_names)
    test_acc_final, per_task = evaluate(model, test_data, n_tasks, task_names, return_per_task=True)
    
    # Output metrics
    print(f"METRIC mean_acc={test_acc_final:.6f}")
    print(f"METRIC train_acc={train_acc_final:.6f}")
    
    # Per-family averages
    from extended_yang19 import GO_TASKS, DM_TASKS, DLYDM_TASKS, MATCH_TASKS
    families = {'go': GO_TASKS, 'dm': DM_TASKS, 'dlydm': DLYDM_TASKS, 'match': MATCH_TASKS}
    for fname, ftasks in families.items():
        accs = [per_task[t] for t in ftasks if t in per_task]
        if accs:
            print(f"METRIC {fname}_acc={np.mean(accs):.6f}")
    
    above_50 = sum(1 for v in per_task.values() if v >= 0.5)
    above_80 = sum(1 for v in per_task.values() if v >= 0.8)
    print(f"METRIC tasks_above_50={above_50}")
    print(f"METRIC tasks_above_80={above_80}")
    
    # Diagnostics
    sorted_tasks = sorted(per_task.items(), key=lambda x: x[1])
    print("\nWorst 10 tasks:", file=sys.stderr)
    for name, acc in sorted_tasks[:10]:
        print(f"  {name}: {acc:.4f}", file=sys.stderr)
    print("\nBest 10 tasks:", file=sys.stderr)
    for name, acc in sorted_tasks[-10:]:
        print(f"  {name}: {acc:.4f}", file=sys.stderr)


def evaluate(model, test_data, n_tasks, task_names, return_per_task=False):
    """Evaluate model on test data. Returns mean accuracy (during decision period)."""
    model.eval()
    
    task_correct = {name: 0 for name in task_names}
    task_total = {name: 0 for name in task_names}
    
    with torch.no_grad():
        for i in range(0, len(test_data), BATCH_SIZE):
            batch = test_data[i:i+BATCH_SIZE]
            obs, gt, mask, rule = collate_batch(batch, n_tasks)
            
            logits = model(obs, rule)
            preds = logits.argmax(dim=-1)
            
            decision_mask = (gt != 0).float() * mask
            correct = (preds == gt).float() * decision_mask
            
            for j, (_, _, tid) in enumerate(batch):
                tname = task_names[tid]
                task_correct[tname] += correct[j].sum().item()
                task_total[tname] += decision_mask[j].sum().item()
    
    per_task = {}
    for name in task_names:
        if task_total[name] > 0:
            per_task[name] = task_correct[name] / task_total[name]
        else:
            per_task[name] = 0.0
    
    mean_acc = np.mean(list(per_task.values()))
    
    if return_per_task:
        return mean_acc, per_task
    return mean_acc


if __name__ == "__main__":
    train()
