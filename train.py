"""Train models on discrete token tasks and neurogym tasks.

Supports:
- Discrete token tasks (associative recall, path integration, navigation)
- Neurogym continuous tasks (extended_yang19.py: go, anti, dm, dms, etc.)

Generalization testing: train on subset of conditions, test on held-out.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, List

import chz
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from dataset import make_dataset_continous, make_neurogym_dataset
from model import make_sequence_model_spec, SequenceModelParams
from extended_yang19 import TASKS as NEUROGYM_TASKS, get_condition_filter

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    # May fail on numpy 2.0 due to np.float_ removal in wandb
    WANDB_AVAILABLE = False
    wandb = None

# Results directory following codebase convention
SCRIPT_DIR = Path(__file__).parent
# RESULTS_DIR = SCRIPT_DIR / "results" / Path(__file__).stem
RESULTS_DIR = SCRIPT_DIR / "results" / "train_2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class WandbLogger:
    """Simple wandb logger similar to zoology."""
    def __init__(self, project: str, config: dict = None):
        if not WANDB_AVAILABLE:
            print("wandb not available, skipping logging")
            self.enabled = False
            return
        self.enabled = True
        self.run = wandb.init(project=project, config=config)
        print(f"Wandb: {self.run.url}")

    def log(self, metrics: dict):
        if not self.enabled:
            return
        wandb.log(metrics)

    def finish(self):
        if not self.enabled:
            return
        self.run.finish()


@chz.chz
class TrainConfig:
    """Training configuration with hyperparameters."""
    # Task selection
    task: str = "go"  # Task name or comma-separated list
    task_group: str = ""  # Task group: 'yang', 'nav', 'all', or empty
    multitask: bool = False  # Train single model on all tasks simultaneously
    model: str = ""  # Model to train. Empty = train all models.

    # Training
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    max_train_steps: Optional[int] = None  # Optional exact optimizer-update budget (stops mid-epoch if reached)

    # Data
    n_train_samples: int = 5000
    n_test_samples: int = 1000

    # Model
    embed_dim: int = 192
    hidden_dim: int = 320

    d_k: int = 64  # KV memory key dim
    d_v: int = 64  # KV memory value dim
    K: str = "1"  # Number of modules: "K" or "K_int,K_mem"
    n_layers: int = 1  # Number of stacked blocks
    residual: bool = False  # Enable residual connections around each block
    shared_weights: bool = False  # Share block weights across layers
    concat_input: bool = False  # Concatenate input embedding with block output at each layer

    # Simulated annealing for selection softmax temperature
    # temperature = tau_base + tau_scale * prev_batch_loss
    # Disabled by default (tau_scale=0 -> temperature=1.0)
    tau_base: float = 0.0   # Base temperature (used when tau_scale > 0)
    tau_scale: float = 0.0  # Scale factor (0 = disabled, use temperature=1.0)

    # Optional routing regularization (for models exposing layer out softmax)
    # Adds to loss:
    #   + routing_entropy_reg * mean_t H(p_t)
    #   + routing_balance_reg * sum_i p_bar_i log(p_bar_i)
    # Positive routing_entropy_reg encourages per-step sparse/peaky routing.
    # Positive routing_balance_reg discourages collapse to one module globally.
    routing_entropy_reg: float = 0.0
    routing_balance_reg: float = 0.0

    # Logging
    eval_interval: int = 1
    log_interval: int = 1

    # Runtime
    num_workers: int = 0  # Parallel workers for data generation
    no_wandb: bool = True  # Disable wandb logging
    wandb_project: str = "module-reuse"  # Wandb project name

    @property
    def K_parsed(self) -> Union[int, Tuple[int, ...]]:
        """Parse K string to int or tuple."""
        if ',' in self.K:
            return tuple(int(x) for x in self.K.split(','))
        return int(self.K)


@dataclass
class NeurogymTaskConfig:
    """Task-specific configuration for neurogym continuous tasks."""
    dim_ring: int = 16
    dt: int = 100
    seq_len: int = 40  # Max ~34 timesteps (DelayMatch tasks), use 40 for safety

    @property
    def input_dim(self):
        """Input dimension: 1 fixation + 2*dim_ring stimulus modalities."""
        return 1 + 2 * self.dim_ring  # 33 for dim_ring=16

    @property
    def output_dim(self):
        """Output dimension: 1 fixation + dim_ring ring choices."""
        return 1 + self.dim_ring  # 17 for dim_ring=16


@dataclass
class TaskConfig:
    """Unified task config for run_task. Supports single and multi-task training."""
    name: str
    input_dim: int
    output_dim: int
    seq_len: int
    input_type: str  # 'continuous' or 'discrete'
    output_group: str  # tasks with same group share output classes
    task_type: str = 'neurogym'  # 'neurogym' or 'nav' - used for data generation dispatch
    condition_filter: Optional[dict] = None  # Filter for held-out conditions, e.g. {'delay': (0, 800)}


# Default neurogym config (shared by all neurogym tasks)
NEUROGYM_CONFIG = NeurogymTaskConfig()


# Default training config
DEFAULT_CONFIG = TrainConfig()


def get_models(vocab_size, config: TrainConfig = DEFAULT_CONFIG, input_type='discrete', input_dim=None, output_dim=None, n_layers=1, routings=None):
    """Model configs (vocab_size set per task).

    Args:
        vocab_size: For discrete tasks, vocabulary size (also output_dim)
        config: Training config with model hyperparameters
        input_type: 'discrete' (embedding) or 'continuous' (projection)
        input_dim: For continuous inputs, input dimension
        output_dim: Output dimension (defaults to vocab_size for discrete)
        n_layers: Number of stacked blocks
        routings: Optional fixed routing vectors per layer for modular model.
                  List of (n_selections,) arrays, one per layer.
    """
    if output_dim is None:
        output_dim = vocab_size
    return {
        "modular": lambda: make_sequence_model_spec(
            "modular",
            input_type=input_type,
            vocab_size=vocab_size,
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            d_k=config.d_k,
            d_v=config.d_v,
            n_layers=n_layers,
            K=config.K_parsed,
            residual=config.residual,
            routings=routings,
            shared_weights=config.shared_weights,
            concat_input=config.concat_input,
        ),
    }


def _generate_task_data(args):
    """Worker function for parallel data generation (picklable)."""
    task_idx, task, seed, n_samples = args

    if task.task_type == 'neurogym':
        factory = NEUROGYM_TASKS[task.name]
        X_raw, Y_raw, task_meta = make_neurogym_dataset(seed, factory, NEUROGYM_CONFIG, n_samples, task.condition_filter)
    elif task.task_type == 'nav':
        from dataset import make_dataset_continous
        X_raw, Y_raw, task_meta = make_dataset_continous(seed, task.name, T=task.seq_len, dt=1, n_samples=n_samples)
    else:
        raise ValueError(f"Unknown task_type: {task.task_type}")

    return task_idx, X_raw, Y_raw, task_meta


def make_dataset(rng, tasks: List[TaskConfig], n_samples_per_task: int, num_workers: int = 0):
    """Generate standardized dataset from task list.

    Tasks with same output_group share output classes.
    Different groups get separate class ranges with offsets.
    Task identity is encoded in input via task one-hot.
    Condition filtering is specified per-task via TaskConfig.condition_filter.

    Args:
        rng: JAX random key
        tasks: List of TaskConfig (each may have condition_filter for held-out conditions)
        n_samples_per_task: Number of samples per task
        num_workers: Number of parallel workers (0 = sequential)

    Returns:
        X: (n_samples, max_seq_len, input_dim) - standardized inputs
        Y: (n_samples, max_seq_len) - targets with class offsets
        meta: list of dicts with task info
        input_dim: total input dimension (max_input_dim + n_tasks)
        total_classes: total number of output classes
    """
    n_tasks = len(tasks)
    max_input_dim = max(t.input_dim for t in tasks)
    max_seq_len = max(t.seq_len for t in tasks)
    input_dim = max_input_dim + n_tasks

    # Compute class offsets per output_group (use max output_dim within each group)
    # e.g., yang(17) → 0, nav(64) → 17
    group_max_output = {}
    for t in tasks:
        if t.output_group not in group_max_output:
            group_max_output[t.output_group] = t.output_dim
        else:
            group_max_output[t.output_group] = max(group_max_output[t.output_group], t.output_dim)

    group_offsets, total_classes = {}, 0
    for group, max_dim in group_max_output.items():
        group_offsets[group] = total_classes
        total_classes += max_dim

    # Get base seed from JAX rng
    base_seed = int(jax.random.randint(rng, (), 0, 2**20))

    # Generate raw data (parallel or sequential)
    task_args = [(idx, t, base_seed + idx, n_samples_per_task) for idx, t in enumerate(tasks)]

    if num_workers > 0:
        from multiprocessing import Pool
        print(f"Generating data with {num_workers} processes...", flush=True)
        with Pool(num_workers) as pool:
            raw_results = pool.map(_generate_task_data, task_args)
    else:
        raw_results = []
        for args in task_args:
            print(f"Generating data for task {args[0] + 1}/{len(tasks)}: {args[1].name}", flush=True)
            raw_results.append(_generate_task_data(args))

    # Process results (pure numpy)
    X_all, Y_all, meta = [], [], []
    for task_idx, X_raw, Y_raw, task_meta in raw_results:
        t = tasks[task_idx]
        offset = group_offsets[t.output_group]

        for i in range(len(X_raw)):
            seq_len = X_raw[i].shape[0]

            # Standardize input dimension
            if t.input_type == 'discrete':
                x = np.eye(max_input_dim, dtype=np.float32)[X_raw[i].squeeze().astype(np.int32)]
            else:
                x = np.pad(X_raw[i], ((0, 0), (0, max_input_dim - t.input_dim)))

            # Append task one-hot
            task_oh = np.zeros((seq_len, n_tasks), dtype=np.float32)
            task_oh[:, task_idx] = 1.0
            x = np.concatenate([x, task_oh], axis=-1)

            # Pad sequence to max_seq_len
            x = np.pad(x, ((0, max_seq_len - seq_len), (0, 0)))

            # Standardize target: shift by class offset, pad with -1
            y = np.where(Y_raw[i] >= 0, Y_raw[i] + offset, Y_raw[i])
            y = np.pad(y, (0, max_seq_len - seq_len), constant_values=-1)

            X_all.append(x)
            Y_all.append(y)
            # Include task_meta fields (e.g., init_state for nav tasks)
            sample_meta = {'task': t.name, 'task_idx': task_idx, 'output_group': t.output_group, 'offset': offset}
            if task_meta and i < len(task_meta):
                sample_meta.update(task_meta[i])
            meta.append(sample_meta)

    # Return numpy arrays (convert to JAX in run_task)
    return np.stack(X_all), np.stack(Y_all), meta, input_dim, total_classes


def compute_accuracy(spec, params, X, Y, meta=None, rng=None, eval_batch_size=256):
    """Compute accuracy on response positions only, in batches to avoid OOM.

    For yang tasks: only count timesteps where target is a direction (Y > 0)
    For discrete tasks: count timesteps where Y >= 0
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    n = X.shape[0]
    total_correct = 0
    total_count = 0

    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        X_batch = jnp.array(X[start:end])
        Y_batch = jnp.array(Y[start:end])
        rng, batch_rng = jax.random.split(rng)
        rngs = jax.random.split(batch_rng, end - start)
        logits = jax.vmap(lambda x, r: spec.apply(params, x, r))(X_batch, rngs)
        preds = jnp.argmax(logits, axis=-1)

        # Build mask based on task type
        if meta is not None:
            masks = []
            for i in range(start, end):
                m = meta[i]
                if m['output_group'] == 'yang':
                    mask_i = Y_batch[i - start] > 0
                else:
                    mask_i = Y_batch[i - start] >= 0
                masks.append(mask_i)
            mask = jnp.stack(masks)
        else:
            mask = Y_batch >= 0

        total_correct += int(jnp.sum((preds == Y_batch) & mask))
        total_count += int(jnp.sum(mask))

    return total_correct / max(total_count, 1)


def batched_loss(spec, params, X, Y, rng, eval_batch_size=256):
    """Compute loss in batches to avoid OOM."""
    n = X.shape[0]
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        rng, batch_rng = jax.random.split(rng)
        batch_loss = float(spec.loss(params, jnp.array(X[start:end]), jnp.array(Y[start:end]), rng=batch_rng))
        total_loss += batch_loss
        n_batches += 1
    return total_loss / max(n_batches, 1)


def plot_nav_predictions(spec, params, X, Y, meta, filepath, task_name, sample_idx=0, n_samples=5):
    """Plot nav task predictions: input sequence, target, and model predictions."""
    x = np.array(X[sample_idx])
    y = np.array(Y[sample_idx])

    # If one-hot encoded (2D), convert back to token indices for visualization
    if x.ndim == 2:
        vocab_size = meta[sample_idx].get('vocab_size', 20) if meta else 20
        x = np.argmax(x[:, :vocab_size], axis=-1)  # Exclude task one-hot

    # Get model predictions
    logits = spec.apply(params, X[sample_idx], jax.random.PRNGKey(sample_idx))
    preds = np.array(jnp.argmax(logits, axis=-1))

    # Adjust for offset
    offset = meta[sample_idx].get('offset', 0)
    y_adj = np.where(y >= 0, y - offset, y)
    preds_adj = np.where(y >= 0, preds - offset, -1)

    # Find query positions (where target != -1)
    query_pos = np.where(y >= 0)[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 6))

    # Plot input sequence
    axes[0].bar(range(len(x)), x, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Token')
    axes[0].set_title(f'{task_name} - Input Sequence')
    axes[0].set_xticks(range(len(x)))

    # Plot target
    axes[1].bar(range(len(y_adj)), y_adj, color='green', alpha=0.7)
    for qp in query_pos:
        axes[1].axvline(qp, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Target')
    axes[1].set_title('Ground Truth (query positions marked)')
    axes[1].set_xticks(range(len(y_adj)))

    # Plot predictions at query positions
    axes[2].bar(range(len(y_adj)), y_adj, color='green', alpha=0.3, label='Target')
    axes[2].bar(range(len(preds_adj)), preds_adj, color='blue', alpha=0.5, label='Prediction')
    for qp in query_pos:
        correct = preds_adj[qp] == y_adj[qp]
        color = 'green' if correct else 'red'
        axes[2].axvline(qp, color=color, linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Token')
    axes[2].set_xlabel('Timestep')
    axes[2].set_title('Predictions vs Target')
    axes[2].legend()
    axes[2].set_xticks(range(len(preds_adj)))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved nav predictions to {filepath}")
    plt.close()

    # Save multiple examples to text
    txt_filepath = filepath.with_suffix('.txt')
    n_samples = min(n_samples, len(X))
    with open(txt_filepath, 'w') as f:
        total_correct = 0
        total_queries = 0
        for i in range(n_samples):
            xi = np.array(X[i])
            yi = np.array(Y[i])

            # Convert one-hot to token indices if needed
            if xi.ndim == 2:
                vocab_size = meta[i].get('vocab_size', 20) if meta else 20
                xi = np.argmax(xi[:, :vocab_size], axis=-1)

            # Get model predictions
            logits_i = spec.apply(params, X[i], jax.random.PRNGKey(i))
            preds_i = np.array(jnp.argmax(logits_i, axis=-1))

            # Adjust for offset
            offset_i = meta[i].get('offset', 0) if meta else 0
            yi_adj = np.where(yi >= 0, yi - offset_i, yi)
            preds_i_adj = np.where(yi >= 0, preds_i - offset_i, -1)

            # Find query positions
            query_pos_i = np.where(yi >= 0)[0]

            f.write(f"=== Example {i+1} ===\n")
            f.write(f"Input:      {[int(v) for v in xi]}\n")
            f.write(f"Target:     {[int(v) for v in yi_adj]}\n")
            f.write(f"Prediction: {[int(v) for v in preds_i_adj]}\n")
            f.write(f"Query pos:  {list(query_pos_i)}\n")
            if len(query_pos_i) > 0:
                correct_i = sum(preds_i_adj[qp] == yi_adj[qp] for qp in query_pos_i)
                total_correct += correct_i
                total_queries += len(query_pos_i)
                f.write(f"Accuracy:   {correct_i}/{len(query_pos_i)} = {correct_i/len(query_pos_i):.1%}\n")
            f.write("\n")

        if total_queries > 0:
            f.write(f"=== Overall ===\n")
            f.write(f"Total accuracy: {total_correct}/{total_queries} = {total_correct/total_queries:.1%}\n")


def count_params(params):
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def extract_gate_means(spec, params, X_samples):
    """Extract mean gate values from samples for tracking over training.

    Args:
        spec: Model spec with apply_with_gates method
        params: Model parameters
        X_samples: Sample inputs to extract gates from (numpy array)

    Returns:
        Dict mapping gate names to mean values.
        Scalar gates return float; vector-valued gates return arrays.
    """
    gate_sums = {}
    n_samples = len(X_samples)

    def process_gate(full_name, values):
        """Process a single gate array and return (name, mean_val)."""
        if not hasattr(values, 'shape'):
            return None

        if values.ndim == 1:
            # For 1D gates like alpha, beta, omega: mean over time
            mean_val = float(jnp.mean(values))
        elif values.ndim == 2:
            # For 2D gates (e.g., 'out' selection weights): mean over seq_len
            mean_val = jnp.mean(values, axis=0)
        else:
            return None

        return mean_val

    for i in range(n_samples):
        _, gates = spec.apply_with_gates(params, jnp.array(X_samples[i]))

        # Handle both flat and nested gate structures.
        for name, values in gates.items():
            if isinstance(values, dict):
                # Nested structure: flatten with layer prefix
                for gate_name, gate_values in values.items():
                    full_name = f'{name}_{gate_name}'
                    mean_val = process_gate(full_name, gate_values)
                    if mean_val is not None:
                        if full_name not in gate_sums:
                            gate_sums[full_name] = 0.0 if isinstance(mean_val, float) else jnp.zeros_like(mean_val)
                        gate_sums[full_name] = gate_sums[full_name] + mean_val
            else:
                # Flat structure
                mean_val = process_gate(name, values)
                if mean_val is not None:
                    if name not in gate_sums:
                        gate_sums[name] = 0.0 if isinstance(mean_val, float) else jnp.zeros_like(mean_val)
                    gate_sums[name] = gate_sums[name] + mean_val

    # Average across samples
    result = {}
    for k, v in gate_sums.items():
        if isinstance(v, float):
            result[k] = v / n_samples
        else:
            result[k] = np.array(v / n_samples)  # Convert to numpy for storage
    return result


def train_model(
    model_name,
    spec,
    X_train,
    Y_train,
    X_test,
    Y_test,
    config: TrainConfig = DEFAULT_CONFIG,
    logger: WandbLogger = None,
    meta_train=None,
    meta_test=None,
    init_params=None,
    track_test_acc_per_task: bool = False,
):
    """Train a single model and return history + final params."""
    rng = jax.random.PRNGKey(config.seed)
    params = init_params if init_params is not None else spec.init(rng)
    print(f"  Parameters: {count_params(params):,}")

    optimizer = optax.adamw(config.lr, weight_decay=2e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def routing_regularization(params, x_batch, temperature):
        """Compute routing regularizers from layer out distributions.

        Returns:
            entropy_penalty: mean temporal entropy H(p_t) across layers/batch
            balance_penalty: mean sum_i p_bar_i log(p_bar_i) across layers/batch
        """
        if spec.apply_with_gates is None:
            zero = jnp.float32(0.0)
            return zero, zero

        n_layers = max(1, config.n_layers)

        def per_sample(x):
            _, gates = spec.apply_with_gates(params, x, temperature=temperature)
            entropy_total = jnp.float32(0.0)
            balance_total = jnp.float32(0.0)

            for layer_idx in range(config.n_layers):
                layer_gates = gates[f"layer{layer_idx}"]
                if "out" not in layer_gates:
                    continue

                # out: (seq_len, n_selections)
                p = jnp.clip(layer_gates["out"], 1e-8, 1.0)
                entropy_total += -jnp.mean(jnp.sum(p * jnp.log(p), axis=-1))

                p_bar = jnp.clip(jnp.mean(p, axis=0), 1e-8, 1.0)
                balance_total += jnp.sum(p_bar * jnp.log(p_bar))

            return entropy_total / n_layers, balance_total / n_layers

        entropy_vals, balance_vals = jax.vmap(per_sample)(x_batch)
        return jnp.mean(entropy_vals), jnp.mean(balance_vals)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch, step_rng, temperature):
        def objective(p):
            loss = spec.loss(p, x_batch, y_batch, rng=step_rng, temperature=temperature)
            if config.routing_entropy_reg != 0.0 or config.routing_balance_reg != 0.0:
                entropy_pen, balance_pen = routing_regularization(p, x_batch, temperature)
                loss = (
                    loss
                    + config.routing_entropy_reg * entropy_pen
                    + config.routing_balance_reg * balance_pen
                )
            return loss

        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'epochs': []}
    history['steps'] = []
    task_eval_indices = None
    if track_test_acc_per_task and meta_test:
        task_eval_indices = {}
        for i, meta in enumerate(meta_test):
            task_name = meta.get("task")
            if task_name is None:
                continue
            task_eval_indices.setdefault(task_name, []).append(i)
        if task_eval_indices:
            history["test_acc_per_task"] = {task_name: [] for task_name in task_eval_indices}

    print(f"\nTraining {model_name}...")
    if config.max_train_steps is not None:
        if config.max_train_steps <= 0:
            raise ValueError(f"max_train_steps must be positive, got {config.max_train_steps}")
        print(f"  Max train steps: {config.max_train_steps}")

    # Simulated annealing setup
    use_annealing = config.tau_scale > 0
    prev_loss = jnp.float32(1.0)
    if use_annealing:
        print(f"  Annealing: tau_base={config.tau_base}, tau_scale={config.tau_scale}")
        history['temperature'] = []
    if config.routing_entropy_reg != 0.0 or config.routing_balance_reg != 0.0:
        print(
            f"  Routing regularization: "
            f"entropy={config.routing_entropy_reg}, balance={config.routing_balance_reg}"
        )

    # Record initial gate stats BEFORE any training (true epoch 0)
    if spec.apply_with_gates is not None:
        history['gate_epochs'] = [0]
        sample_gates = extract_gate_means(spec, params, X_test[:3])
        for gate_name, gate_value in sample_gates.items():
            history[gate_name] = [gate_value]

    global_step = 0
    temperature = jnp.float32(1.0)

    for epoch in range(config.n_epochs):
        rng, shuffle_rng = jax.random.split(rng)
        # Keep permutation as NumPy array to avoid loading full dataset to GPU
        perm = np.array(jax.random.permutation(shuffle_rng, len(X_train)))
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]

        stop_requested = False
        for i in range(0, len(X_train), config.batch_size):
            rng, step_rng = jax.random.split(rng)
            # Convert batch to JAX arrays just before use
            x_batch = jnp.array(X_shuf[i:i+config.batch_size])
            y_batch = jnp.array(Y_shuf[i:i+config.batch_size])
            # Compute temperature from previous batch loss
            temperature = jnp.float32(config.tau_base + config.tau_scale * prev_loss) if use_annealing else jnp.float32(1.0)
            params, opt_state, batch_loss = train_step(
                params, opt_state, x_batch, y_batch, step_rng, temperature
            )
            if use_annealing:
                prev_loss = batch_loss
            global_step += 1
            if config.max_train_steps is not None and global_step >= config.max_train_steps:
                stop_requested = True
                break

        if stop_requested or epoch % config.eval_interval == 0 or epoch == config.n_epochs - 1:
            rng, eval_rng1, eval_rng2, acc_rng1, acc_rng2 = jax.random.split(rng, 5)
            # Batched evaluation to avoid OOM
            train_loss = batched_loss(spec, params, X_train[:500], Y_train[:500], rng=eval_rng1)
            test_loss = batched_loss(spec, params, X_test, Y_test, rng=eval_rng2)
            meta_train_slice = meta_train[:500] if meta_train else None
            train_acc = compute_accuracy(spec, params, X_train[:500], Y_train[:500], meta_train_slice, rng=acc_rng1)
            test_acc = compute_accuracy(spec, params, X_test, Y_test, meta_test, rng=acc_rng2)

            history['epochs'].append(epoch)
            history['steps'].append(global_step)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            if task_eval_indices:
                for task_idx, (task_name, idxs) in enumerate(task_eval_indices.items()):
                    idx_arr = np.array(idxs)
                    task_acc = compute_accuracy(
                        spec,
                        params,
                        jnp.array(X_test[idx_arr]),
                        jnp.array(Y_test[idx_arr]),
                        [meta_test[i] for i in idxs],
                        rng=jax.random.fold_in(acc_rng2, task_idx + 1),
                    )
                    history["test_acc_per_task"][task_name].append(float(task_acc))

            # Track temperature when annealing is enabled
            if use_annealing:
                cur_temp = float(temperature)
                history['temperature'].append(cur_temp)

            # Log to wandb
            if logger:
                log_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    f'{model_name}/train_loss': train_loss,
                    f'{model_name}/test_loss': test_loss,
                    f'{model_name}/train_acc': train_acc,
                    f'{model_name}/test_acc': test_acc,
                }
                if use_annealing:
                    log_dict[f'{model_name}/temperature'] = float(temperature)
                logger.log(log_dict)

            # Track gate values for models with apply_with_gates (every 10 epochs to save time/memory)
            # Skip epoch 0 since initial state was recorded before training loop
            if spec.apply_with_gates is not None and (
                stop_requested or (epoch % 10 == 0 and epoch > 0) or epoch == config.n_epochs - 1
            ):
                history['gate_epochs'].append(epoch)
                sample_gates = extract_gate_means(spec, params, X_test[:3])
                for gate_name, gate_value in sample_gates.items():
                    if gate_name not in history:
                        history[gate_name] = []
                    history[gate_name].append(gate_value)

            if stop_requested or epoch % config.log_interval == 0 or epoch == config.n_epochs - 1:
                temp_str = f", temp={float(temperature):.4f}" if use_annealing else ""
                print(f"  Epoch {epoch:3d} (step {global_step:5d}): train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, "
                      f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}{temp_str}")

        if stop_requested:
            break

    return history, params


def plot_training_curves(results, filepath):
    """Plot training curves (loss and accuracy) for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    colors = plt.cm.tab10.colors

    for i, (model_name, history) in enumerate(results.items()):
        color = colors[i % len(colors)]
        epochs = history['epochs']

        axes[0].plot(epochs, history['train_loss'], '-', color=color, label=f'{model_name} train')
        axes[0].plot(epochs, history['test_loss'], '--', color=color, label=f'{model_name} test')

        axes[1].plot(epochs, history['train_acc'], '-', color=color, label=f'{model_name} train')
        axes[1].plot(epochs, history['test_acc'], '--', color=color, label=f'{model_name} test')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_yscale('log')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved training curves to {filepath}")
    plt.close()


def plot_module_gates_over_training(history, filepath, K=1, n_layers=1):
    """Plot module gates (alpha, beta, omega) over training epochs.

    Args:
        history: Training history dict containing gate values
        filepath: Path to save the plot
        K: Number of modules per type
        n_layers: Number of layers
    """
    # Use gate_epochs if available (gates tracked less frequently), else epochs
    if 'gate_epochs' not in history and 'epochs' not in history:
        return

    epochs = history.get('gate_epochs', history.get('epochs'))

    # Check for gate keys - modular model uses layer{l}_int{k}_alpha format
    def get_key(layer, module_type, idx, gate):
        """Get gate key from history."""
        key = f'layer{layer}_{module_type}{idx}_{gate}'
        if key in history:
            return key
        return None

    # Check if any gate data exists
    has_int_gates = get_key(0, 'int', 0, 'alpha') is not None
    has_mem_gates = get_key(0, 'mem', 0, 'omega') is not None

    if not has_int_gates and not has_mem_gates:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Integrator gates (alpha, beta)
    for l in range(n_layers):
        for k in range(K):
            alpha_key = get_key(l, 'int', k, 'alpha')
            beta_key = get_key(l, 'int', k, 'beta')
            if alpha_key and alpha_key in history and history[alpha_key]:
                axes[0].plot(epochs, history[alpha_key], '-o',
                            label=f'L{l} Int{k} α', markersize=3)
            if beta_key and beta_key in history and history[beta_key]:
                axes[0].plot(epochs, history[beta_key], '-s',
                            label=f'L{l} Int{k} β', markersize=3)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Gate Value')
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[0].set_title('Integrator Gates')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: Memory gates (omega)
    for l in range(n_layers):
        for k in range(K):
            omega_key = get_key(l, 'mem', k, 'omega')
            if omega_key and omega_key in history and history[omega_key]:
                axes[1].plot(epochs, history[omega_key], '-^',
                            label=f'L{l} Mem{k} ω', markersize=3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Gate Value')
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    axes[1].set_title('Memory Gates')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved module gates plot to {filepath}")
    plt.close()


def plot_output_selection_over_training(history, filepath, n_layers=1):
    """Plot output selection weights over training epochs.

    Args:
        history: Training history dict containing output selection weights
        filepath: Path to save the plot
        n_layers: Number of layers
    """
    # Use gate_epochs if available (gates tracked less frequently), else epochs
    if 'gate_epochs' not in history and 'epochs' not in history:
        return

    epochs = history.get('gate_epochs', history.get('epochs'))

    has_modular_out = 'layer0_out' in history

    if not has_modular_out:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors

    # Modular model: layer{l}_out is array of shape (n_epochs, n_selections)
    for l in range(n_layers):
        key = f'layer{l}_out'
        if key in history and history[key]:
            weights_over_time = np.array(history[key])
            n_selections = weights_over_time.shape[1]
            for out_idx in range(n_selections):
                label = f'L{l} Int' if out_idx == 0 else f'L{l} Mem' if out_idx == 1 else f'L{l} Null'
                marker = '-o' if out_idx == 0 else '-s' if out_idx == 1 else '-^'
                ax.plot(
                    epochs,
                    weights_over_time[:, out_idx],
                    marker,
                    label=label,
                    markersize=3,
                    color=colors[(l * 3 + out_idx) % len(colors)],
                )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Selection Weight')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2)
    ax.set_title('Output Selection')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved output selection plot to {filepath}")
    plt.close()


def plot_generalization_bars(all_accs, filepath, condition_labels=None):
    """Plot bar chart comparing train/test/extrapolation accuracy per model.

    Style matches generalization_analysis_rnn_vs_rnn_integrator_hybrid.py
    """
    models = list(all_accs.keys())
    conditions = list(all_accs[models[0]].keys())
    n_models = len(models)
    n_conds = len(conditions)

    # Default condition labels
    if condition_labels is None:
        condition_labels = {
            'train': 'Training',
            'test': 'Test (same dist)',
            'extrap': 'Extrapolation',
        }

    # Colors matching original script
    condition_colors = {
        'train': '#4C72B0',
        'test': '#55A868',
        'extrap': '#C44E52',
    }

    base_positions = np.arange(n_models, dtype=float)
    bar_width = 0.2 if n_conds > 1 else 0.5

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2), dpi=150)

    for idx, cond in enumerate(conditions):
        vals = [all_accs[m][cond] for m in models]
        offsets = base_positions + (idx - (n_conds - 1) / 2) * bar_width
        label = condition_labels.get(cond, cond)
        color = condition_colors.get(cond, plt.cm.tab10.colors[idx])

        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            label=label,
            color=color,
            alpha=0.85,
        )
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel('Accuracy')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    print(f"Saved generalization plot to {filepath}")
    plt.close(fig)


def plot_train_test_acc_by_task(spec, params, X_train, Y_train, meta_train, X_test, Y_test, meta_test, task_names, output_path, model_name=""):
    """Plot train/test accuracy by task, ordered by test accuracy."""
    per_task_acc_rows = []
    rng = jax.random.PRNGKey(42)
    for task_name in task_names:
        train_idx = [i for i, m in enumerate(meta_train) if m['task'] == task_name]
        test_idx = [i for i, m in enumerate(meta_test) if m['task'] == task_name]
        if not train_idx or not test_idx:
            continue
        rng, tr_rng, te_rng = jax.random.split(rng, 3)
        train_acc = float(compute_accuracy(
            spec, params,
            jnp.array(X_train[jnp.array(train_idx)]),
            jnp.array(Y_train[jnp.array(train_idx)]),
            [meta_train[i] for i in train_idx],
            rng=tr_rng,
        ))
        test_acc = float(compute_accuracy(
            spec, params,
            jnp.array(X_test[jnp.array(test_idx)]),
            jnp.array(Y_test[jnp.array(test_idx)]),
            [meta_test[i] for i in test_idx],
            rng=te_rng,
        ))
        per_task_acc_rows.append({"task": task_name, "train_acc": train_acc, "test_acc": test_acc})

    if not per_task_acc_rows:
        return

    rows_sorted = sorted(per_task_acc_rows, key=lambda r: r["test_acc"], reverse=True)
    task_labels = [r["task"] for r in rows_sorted]
    train_vals = np.array([r["train_acc"] for r in rows_sorted], dtype=np.float32)
    test_vals = np.array([r["test_acc"] for r in rows_sorted], dtype=np.float32)

    x = np.arange(len(task_labels))
    width = 0.42
    fig_w = max(8.0, 0.45 * len(task_labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar(x - width / 2, train_vals, width=width, label="Train", color="#1f77b4")
    ax.bar(x + width / 2, test_vals, width=width, label="Test", color="#ff7f0e")

    ax.set_xlabel("Task (sorted by test acc)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    title = f"Per-Task Train/Test Accuracy ({model_name})" if model_name else "Per-Task Train/Test Accuracy"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def run_task(tasks: List[TaskConfig], config: TrainConfig = DEFAULT_CONFIG, model_name: Optional[str] = None, logger: WandbLogger = None, results_dir: Optional[Path] = None, num_workers: int = 0, test_tasks: Optional[List[TaskConfig]] = None, routings: Optional[List] = None):
    """Train on task(s). Works for single or multiple tasks.

    Args:
        tasks: List of TaskConfig objects (for training)
        config: Training configuration
        model_name: If specified, only train this model
        logger: Optional wandb logger
        results_dir: Directory for results
        num_workers: Number of parallel workers for data generation (0 = sequential)
        test_tasks: Optional separate task configs for testing (e.g., with held-out conditions).
                    If None, uses same tasks as training.
        routings: Optional fixed routing vectors per layer for modular model.
                  List of (n_selections,) arrays, one per layer.
    """
    # Use test_tasks if provided, otherwise use same as train
    if test_tasks is None:
        test_tasks = tasks

    n_per_task = config.n_train_samples // len(tasks)

    # Dataset caching: save/load from disk to avoid regenerating each run
    cache_dir = SCRIPT_DIR / "cached_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib, json as _json, pickle
    cache_key_data = _json.dumps({
        "train_tasks": [(t.name, t.condition_filter) for t in tasks],
        "test_tasks": [(t.name, t.condition_filter) for t in test_tasks],
        "n_per_task": n_per_task,
        "n_test_per_task": config.n_test_samples // len(test_tasks),
    }, sort_keys=True)
    cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"dataset_{cache_hash}.pkl"

    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        X_train, Y_train, meta_train = cached["X_train"], cached["Y_train"], cached["meta_train"]
        X_test, Y_test, meta_test = cached["X_test"], cached["Y_test"], cached["meta_test"]
        input_dim, output_dim = cached["input_dim"], cached["output_dim"]
    else:
        X_train, Y_train, meta_train, input_dim, output_dim = make_dataset(
            jax.random.PRNGKey(0), tasks, n_per_task, num_workers=num_workers
        )
        X_test, Y_test, meta_test, _, _ = make_dataset(
            jax.random.PRNGKey(1), test_tasks, config.n_test_samples // len(test_tasks), num_workers=num_workers
        )
        print(f"Caching dataset to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X_train": X_train, "Y_train": Y_train, "meta_train": meta_train,
                "X_test": X_test, "Y_test": Y_test, "meta_test": meta_test,
                "input_dim": input_dim, "output_dim": output_dim,
            }, f)

    # Keep datasets as NumPy arrays - only convert batches to JAX during training
    # X_train, Y_train = jnp.array(X_train), jnp.array(Y_train)
    # X_test, Y_test = jnp.array(X_test), jnp.array(Y_test)

    task_names = [t.name for t in tasks]
    print(f"\n{'='*60}")
    print(f"Tasks: {task_names}")
    print(f"input_dim={input_dim}, output_dim={output_dim}")
    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"Config: n_epochs={config.n_epochs}, lr={config.lr}, batch_size={config.batch_size}")
    print(f"{'='*60}")

    # Create results directory
    base_dir = results_dir if results_dir is not None else RESULTS_DIR
    if len(tasks) == 1:
        task_results_dir = base_dir / tasks[0].name
    else:
        task_results_dir = base_dir / "multitask"
    task_results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import json
    # config_path = task_results_dir / "config.json"
    # with open(config_path, 'w') as f:
    #     json.dump(chz.asdict(config), f, indent=2)
    # print(f"Saved config to {config_path}")

    # Get models
    models = get_models(
        vocab_size=output_dim,
        config=config,
        input_type='continuous',  # always continuous after standardization
        input_dim=input_dim,
        output_dim=output_dim,
        n_layers=config.n_layers,
        routings=routings
    )
    if model_name:
        models = {model_name: models[model_name]}

    # Train all models
    results = {}
    all_accs = {}

    for name, make_spec in models.items():
        spec = make_spec()
        model_results_dir = task_results_dir / name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        # save config in model_results_dir
        config_path = model_results_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(chz.asdict(config), f, indent=2)
        print(f"Saved config to {config_path}")

        init_params = spec.init(jax.random.PRNGKey(config.seed))
        history, params = train_model(
            name,
            spec,
            X_train,
            Y_train,
            X_test,
            Y_test,
            config,
            logger=logger,
            meta_train=meta_train,
            meta_test=meta_test,
            init_params=init_params,
        )
        results[name] = (spec, params, history)

        # Evaluate (accuracy computed on response period only for yang tasks)
        acc_rng1, acc_rng2 = jax.random.split(jax.random.PRNGKey(0), 2)
        all_accs[name] = {
            'train': compute_accuracy(spec, params, X_train[:500], Y_train[:500], meta_train[:500], rng=acc_rng1),
            'test': compute_accuracy(spec, params, X_test, Y_test, meta_test, rng=acc_rng2),
        }

        # Plot module gates and output selection over training for models with apply_with_gates
        if spec.apply_with_gates is not None:
            K_val = config.K_parsed if isinstance(config.K_parsed, int) else config.K_parsed[0]
            plot_module_gates_over_training(
                history, model_results_dir / 'module_gates_over_training.png',
                K=K_val, n_layers=config.n_layers
            )
            plot_output_selection_over_training(
                history, model_results_dir / 'output_selection_over_training.png',
                n_layers=config.n_layers
            )

        plot_training_curves({name: history}, model_results_dir / 'training_curves.png')
        plot_generalization_bars({name: all_accs[name]}, model_results_dir / 'generalization.png')

        # Per-task accuracy bar chart (sorted by test acc)
        if len(tasks) > 1:
            plot_train_test_acc_by_task(
                spec, params,
                X_train, Y_train, meta_train,
                X_test, Y_test, meta_test,
                task_names,
                model_results_dir / 'train_test_acc_by_task_sorted.png',
                model_name=name,
            )

        # Plot predictions per task based on task type
        for task in tasks:
            if task.output_group == 'yang':
                # Yang tasks: use standard predictions plot
                yang_train_indices = [i for i, m in enumerate(meta_train) if m['task'] == task.name]
                yang_test_indices = [i for i, m in enumerate(meta_test) if m['task'] == task.name]
                if yang_train_indices:
                    pred_dir = model_results_dir / 'predictions_train'
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    # Convert sliced data to JAX arrays for plotting
                    X_task = jnp.array(X_train[jnp.array(yang_train_indices[:3])])
                    Y_task = jnp.array(Y_train[jnp.array(yang_train_indices[:3])])
                    meta_task = [meta_train[i] for i in yang_train_indices[:3]]
                    plot_predictions(spec, params, X_task, Y_task, meta_task, pred_dir / f'{task.name}.png', n_samples=3)
                if yang_test_indices:
                    pred_dir = model_results_dir / 'predictions_test'
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    # Convert sliced data to JAX arrays for plotting
                    X_task = jnp.array(X_test[jnp.array(yang_test_indices[:3])])
                    Y_task = jnp.array(Y_test[jnp.array(yang_test_indices[:3])])
                    meta_task = [meta_test[i] for i in yang_test_indices[:3]]
                    plot_predictions(spec, params, X_task, Y_task, meta_task, pred_dir / f'{task.name}.png', n_samples=3)

            elif task.output_group == 'nav':
                # Nav tasks: save sequence prediction diagnostics.
                nav_train_indices = [i for i, m in enumerate(meta_train) if m['task'] == task.name]
                nav_test_indices = [i for i, m in enumerate(meta_test) if m['task'] == task.name]

                # Train predictions
                if nav_train_indices:
                    pred_dir = model_results_dir / 'predictions_train'
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    # Filter to nav task data for multi-example text output
                    X_nav = jnp.array(X_train[jnp.array(nav_train_indices)])
                    Y_nav = jnp.array(Y_train[jnp.array(nav_train_indices)])
                    meta_nav = [meta_train[i] for i in nav_train_indices]
                    plot_nav_predictions(spec, params, X_nav, Y_nav, meta_nav,
                                        pred_dir / f'{task.name}_predictions.png',
                                        task.name, sample_idx=0, n_samples=20)

                # Test predictions
                if nav_test_indices:
                    pred_dir = model_results_dir / 'predictions_test'
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    # Filter to nav task data for multi-example text output
                    X_nav = jnp.array(X_test[jnp.array(nav_test_indices)])
                    Y_nav = jnp.array(Y_test[jnp.array(nav_test_indices)])
                    meta_nav = [meta_test[i] for i in nav_test_indices]
                    plot_nav_predictions(spec, params, X_nav, Y_nav, meta_nav,
                                        pred_dir / f'{task.name}_predictions.png',
                                        task.name, sample_idx=0, n_samples=20)

    # Combined plots
    if len(results) > 1:
        # Extract histories for plotting
        histories = {name: hist for name, (_, _, hist) in results.items()}
        plot_training_curves(histories, task_results_dir / 'training_curves_all_models.png')
        plot_generalization_bars(all_accs, task_results_dir / 'generalization_all_models.png')

    # Print final results
    print(f"\nFinal Results:")
    print("-" * 60)
    for name in models.keys():
        print(f"{name:15s}: train={all_accs[name]['train']:.2%}, test={all_accs[name]['test']:.2%}")

    return results, all_accs


def plot_predictions(spec, params, X, Y, meta, filepath, n_samples=5):
    """Plot model predictions vs ground truth for individual samples.

    Shows observation heatmap, ground truth, and model predictions.
    """
    n_samples = min(n_samples, len(X))
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        x = X[i]
        y = Y[i]
        task_info = meta[i]

        # Get model predictions
        logits = spec.apply(params, x, jax.random.PRNGKey(i))
        preds = jnp.argmax(logits, axis=-1)

        # Adjust for class offset
        offset = task_info.get('offset', 0)
        y_adjusted = jnp.where(y >= 0, y - offset, y)
        preds_adjusted = jnp.where(y >= 0, preds - offset, -1)

        # Find valid timesteps (where target is not -1)
        valid_mask = y >= 0
        seq_len = int(jnp.sum(valid_mask)) if jnp.any(valid_mask) else x.shape[0]

        # Plot observation (input without task one-hot)
        ax = axes[i, 0]
        n_tasks = len(set(m['task_idx'] for m in meta))
        obs_dim = x.shape[-1] - n_tasks  # Remove task one-hot
        ax.imshow(x[:seq_len, :obs_dim].T, aspect='auto', origin='lower')
        ax.set_ylabel('Obs dim')
        ax.set_title(f"{task_info['task']} - Observation")

        # Plot ground truth
        ax = axes[i, 1]
        ax.plot(np.array(y_adjusted[:seq_len]), 'g-', label='Ground Truth', linewidth=2)
        ax.set_ylabel('Class')
        ax.set_title('Ground Truth')
        ax.legend()

        # Plot predictions vs ground truth
        ax = axes[i, 2]
        ax.plot(np.array(y_adjusted[:seq_len]), 'g-', label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(np.array(preds_adjusted[:seq_len]), 'b--', label='Prediction', linewidth=2, alpha=0.7)
        ax.set_ylabel('Class')
        ax.set_xlabel('Timestep')
        # Compute accuracy using the same function as training
        sample_acc = compute_accuracy(spec, params, x[None], y[None], [task_info], rng=jax.random.PRNGKey(i))
        ax.set_title(f'Predictions (acc: {sample_acc:.1%})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Saved predictions to {filepath}")
    plt.close()


def make_neurogym_task(name: str, condition_filter: Optional[dict] = None) -> TaskConfig:
    """Create TaskConfig for a neurogym task.
    
    Args:
        name: Task name from NEUROGYM_TASKS
        condition_filter: Optional filter for held-out conditions, e.g.:
            {'delay': (0, 800)} - keep delays in range [0, 800]
            {'delay': [0, 100, 200]} - keep only these specific delays
            {'ground_truth': [0, 1, 2]} - keep only these directions
    """
    cfg = NEUROGYM_CONFIG
    return TaskConfig(
        name=name,
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        seq_len=cfg.seq_len,
        input_type='continuous',
        output_group='yang',
        task_type='neurogym',
        condition_filter=condition_filter
    )


def make_neurogym_task_with_split(name: str, split: str = 'train') -> TaskConfig:
    """Create TaskConfig with default condition split for a neurogym task.
    
    Uses get_condition_filter() to apply default splits:
    - For delay tasks: train on delays 0-800, test on 900-1100
    - For other tasks: no filtering
    
    Args:
        name: Task name from NEUROGYM_TASKS
        split: 'train', 'test', or 'all'
    
    Returns:
        TaskConfig with appropriate condition_filter
    """
    condition_filter = get_condition_filter(name, split)
    return make_neurogym_task(name, condition_filter=condition_filter)


# Navigation task configs
# Token layout for *_4actions tasks: actions [0-3], positions [4-19], stimuli [20-35]
NAV_TASKS = {
    'path_integration_4actions': {'seq_len': 16, 'vocab_size': 4 + 16 + 16},  # 36
    'navigation_4actions': {'seq_len': 16, 'vocab_size': 4 + 16 + 16},  # 36
    'path_integration': {'seq_len': 16, 'vocab_size': 32},
    'navigation': {'seq_len': 64, 'vocab_size': 64},
    'associative_recall': {'seq_len': 16, 'vocab_size': 64},
}


def make_nav_task(name: str) -> TaskConfig:
    """Create TaskConfig for a navigation/memory task."""
    cfg = NAV_TASKS[name]
    return TaskConfig(
        name=name,
        input_dim=cfg['vocab_size'],  # one-hot encoding
        output_dim=cfg['vocab_size'],
        seq_len=cfg['seq_len'],
        input_type='discrete',
        output_group='nav',
        task_type='nav'
    )


def make_task_config(name: str, split: str = 'train') -> TaskConfig:
    """Create TaskConfig for any task (nav or neurogym) with default split.
    
    Args:
        name: Task name
        split: 'train', 'test', or 'all' (for neurogym delay tasks)
    """
    if name in NAV_TASKS:
        return make_nav_task(name)
    elif name in NEUROGYM_TASKS:
        return make_neurogym_task_with_split(name, split)
    else:
        raise ValueError(f"Unknown task: {name}. Available: {list(NAV_TASKS.keys()) + list(NEUROGYM_TASKS.keys())}")


def main(config: TrainConfig):
    """Main entry point for training."""
    # Initialize wandb logger
    logger = None
    if not config.no_wandb:
        wandb_config = chz.asdict(config)
        logger = WandbLogger(project=config.wandb_project, config=wandb_config)

    # Build task lists (train and test splits)
    task_names = []
    if config.task_group:
        for group in config.task_group.split(','):
            group = group.strip()
            if group == 'yang' or group == 'all':
                task_names.extend(NEUROGYM_TASKS.keys())
            if group == 'nav' or group == 'all':
                task_names.extend(NAV_TASKS.keys())
    elif config.task:
        # Parse comma-separated task names
        task_names = [t.strip() for t in config.task.split(',')]
    else:
        task_names = [config.task]

    # Create train and test task configs
    train_tasks = [make_task_config(name, 'train') for name in task_names]
    test_tasks = [make_task_config(name, 'test') for name in task_names]

    # Model name (empty string means train all)
    model_name = config.model if config.model else None

    if config.multitask:
        # Train single model on all tasks simultaneously
        run_task(train_tasks, config, model_name=model_name, logger=logger, num_workers=config.num_workers, test_tasks=test_tasks)
    else:
        # Train on each task separately
        all_results = {}
        for train_task, test_task in zip(train_tasks, test_tasks):
            results, accs = run_task([train_task], config, model_name=model_name, logger=logger, num_workers=config.num_workers, test_tasks=[test_task])
            all_results[train_task.name] = (results, accs)

        # Print summary
        if len(train_tasks) > 1:
            print("\n" + "=" * 60)
            print("Summary (all tasks)")
            print("=" * 60)
            for task_name, (results, accs) in all_results.items():
                print(f"\n{task_name}:")
                for model_name, cond_accs in accs.items():
                    acc_str = ", ".join(f"{k}={v:.2%}" for k, v in cond_accs.items())
                    print(f"  {model_name:15s}: {acc_str}")

    # Finish wandb
    if logger:
        logger.finish()


if __name__ == "__main__":
    config = chz.entrypoint(TrainConfig)
    main(config)
