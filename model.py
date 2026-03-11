"""Unified modular model implementation for autoneuro.

Contains:
- ModelSpec (minimal model interface)
- Modular block implementation
- Sequence wrapper + make_sequence_model_spec
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct

class ModelSpec:
    """Minimal descriptor for a trainable model."""

    def __init__(self, init, apply, loss, apply_with_gates=None):
        self.init = init
        self.apply = apply
        self.loss = loss
        self.apply_with_gates = apply_with_gates  # Optional: for models that expose gates


# =============================================================================
# Module 1: Integrator
# =============================================================================

@struct.dataclass
class IntegratorParams:
    """Parameters for integrator module with lateral recurrence.

    h_{t+1} = (1-α_t) h_t + α_t (W_1 x_t + b_1) + β_t (W_2 x_t + b_2) + W_rec tanh(h_t)
    z_t = [sin(h_t * freqs), cos(h_t * freqs)]
    y_t = W_out @ z_t

    W_rec provides lateral recurrent connections (inspired by cortical circuits).
    """
    W1: jnp.ndarray              # (h_dim, input_dim)
    b1: jnp.ndarray              # (h_dim,)
    W2: jnp.ndarray              # (h_dim, input_dim)
    b2: jnp.ndarray              # (h_dim,)
    W_rec: jnp.ndarray           # (h_dim, h_dim) - lateral recurrence
    w_alpha: jnp.ndarray         # (input_dim,)
    b_alpha: jnp.ndarray         # (1,)
    w_beta: jnp.ndarray          # (input_dim,)
    b_beta: jnp.ndarray          # (1,)
    W_out: jnp.ndarray           # (output_dim, 2*h_dim*num_freqs)
    freqs: jnp.ndarray           # (num_freqs,)


def integrator_step(params: IntegratorParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of integrator module.

    Args:
        params: IntegratorParams
        state: h - hidden state (h_dim,)
        x: input (input_dim,)

    Returns:
        h_new: updated hidden state (h_dim,)
        y: output (output_dim,)
        alpha: alpha gate value (scalar)
        beta: beta gate value (scalar)
    """
    h = state

    # Gates
    alpha = jax.nn.sigmoid(params.w_alpha @ x + params.b_alpha).squeeze()
    beta = jax.nn.sigmoid(params.w_beta @ x + params.b_beta).squeeze()

    # Integrator update (no lateral recurrence)
    h_new = (1 - alpha) * h + alpha * (params.W1 @ x + params.b1) + beta * (params.W2 @ x + params.b2)

    # Multi-frequency positional encoding
    h_scaled = h_new[:, None] * params.freqs[None, :]  # (h_dim, num_freqs)
    z = jnp.concatenate([jnp.sin(h_scaled).flatten(), jnp.cos(h_scaled).flatten()])

    # Output
    y = params.W_out @ z

    return h_new, y, alpha, beta


# =============================================================================
# Module 2: Memory
# =============================================================================

@struct.dataclass
class MemoryParams:
    """Parameters for memory module with synaptic decay.

    k_t = normalize(W_k x_t)
    v_t = W_v x_t, q_t = normalize(W_q x_t)
    δ_t = σ(w_decay @ x_t + b_decay)  (decay/forget gate)
    S_t = δ_t * S_{t-1} - ω_t (S_{t-1} k_t - v_t) k_t^T
    m_t = S_t q_t
    y_t = W_out @ m_t
    """
    W_k: jnp.ndarray             # (d_k, input_dim)
    W_v: jnp.ndarray             # (d_v, input_dim)
    W_q: jnp.ndarray             # (d_k, input_dim)
    w_omega: jnp.ndarray         # (input_dim,)
    b_omega: jnp.ndarray         # (1,)
    w_decay: jnp.ndarray         # (input_dim,) - synaptic decay gate
    b_decay: jnp.ndarray         # (1,)
    W_out: jnp.ndarray           # (output_dim, d_v)


def memory_step(params: MemoryParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of memory module.

    Args:
        params: MemoryParams
        state: S - memory matrix (d_v, d_k)
        x: input (input_dim,)

    Returns:
        S_new: updated memory matrix (d_v, d_k)
        y: output (output_dim,)
        omega: omega gate value (scalar)
    """
    S = state

    # Key, value, query from x
    k = params.W_k @ x
    k = k / jnp.sqrt(jnp.sum(k**2) + 1e-6)  # normalize

    v = params.W_v @ x

    q = params.W_q @ x
    q = q / jnp.sqrt(jnp.sum(q**2) + 1e-6)  # normalize

    # Write gate
    omega = jax.nn.sigmoid(params.w_omega @ x + params.b_omega).squeeze()

    # Synaptic decay gate (inspired by synaptic depression)
    decay = jax.nn.sigmoid(params.w_decay @ x + params.b_decay).squeeze()

    # Delta rule update with decay
    S_decayed = decay * S
    error = S_decayed @ k - v
    S_new = S_decayed - omega * jnp.outer(error, k)

    # Read
    m = S_new @ q

    # Output
    y = params.W_out @ m

    return S_new, y, omega


# =============================================================================
# Module 3: Match/Mismatch Detector (Hippocampal/PFC-inspired)
# =============================================================================

@struct.dataclass
class OscillatorParams:
    """Parameters for match/mismatch detector module.

    Inspired by hippocampal match enhancement / mismatch suppression
    and PFC comparison circuits. Stores a reference pattern (gated write)
    and computes match (element-wise product) and mismatch (element-wise
    difference) between current input projection and stored reference.

    ref_t = (1-w_t) * ref_{t-1} + w_t * (W_enc x_t + b_enc)
    probe = W_probe x_t + b_probe
    match = probe ⊙ ref_t       (element-wise match)
    mismatch = (probe - ref_t)^2  (squared difference)
    z_t = [match, mismatch]
    y_t = W_out @ z_t
    """
    W_enc: jnp.ndarray           # (h_dim, input_dim) - encode input to reference
    b_enc: jnp.ndarray           # (h_dim,)
    W_probe: jnp.ndarray         # (h_dim, input_dim) - encode input to probe
    b_probe: jnp.ndarray         # (h_dim,)
    w_gamma: jnp.ndarray         # (input_dim,) - write gate for reference update
    b_gamma: jnp.ndarray         # (1,)
    W_out: jnp.ndarray           # (output_dim, 2*h_dim)


def oscillator_step(params: OscillatorParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of match/mismatch detector.

    Args:
        params: OscillatorParams (reusing name for compatibility)
        state: ref - stored reference pattern (h_dim,)
        x: input (input_dim,)

    Returns:
        ref_new: updated reference (h_dim,)
        y: output (output_dim,)
        gamma: write gate value (scalar)
    """
    ref = state

    # Write gate: controls reference update
    gamma = jax.nn.sigmoid(params.w_gamma @ x + params.b_gamma).squeeze()

    # Encode input to reference and probe spaces
    encoded = jnp.tanh(params.W_enc @ x + params.b_enc)
    probe = jnp.tanh(params.W_probe @ x + params.b_probe)

    # Update reference (gated leaky integration)
    ref_new = (1 - gamma) * ref + gamma * encoded

    # Match: element-wise product (high when similar)
    match = probe * ref_new

    # Mismatch: squared difference (high when different)
    mismatch = (probe - ref_new) ** 2

    # Concatenate match and mismatch signals
    z = jnp.concatenate([match, mismatch])

    # Output projection
    y = params.W_out @ z

    return ref_new, y, gamma


# =============================================================================
# Module 4: Gated Working Memory (Prefrontal cortex-inspired)
# =============================================================================

@struct.dataclass
class GatedMemoryParams:
    """Parameters for gated working memory module.

    Inspired by prefrontal cortex gating theories (O'Reilly & Frank).
    Separate write, forget, and read gates control access to a
    maintained hidden state. Unlike LSTM, uses explicit forget and
    content-based read gating.

    f_t = σ(w_f @ x_t + b_f)           (forget gate)
    w_t = σ(w_w @ x_t + b_w)           (write gate)
    c_t = f_t * c_{t-1} + w_t * tanh(W_c x_t + b_c)  (cell state)
    r_t = σ(w_r @ x_t + b_r)           (read gate)
    y_t = W_out @ (r_t * tanh(c_t))
    """
    W_c: jnp.ndarray             # (h_dim, input_dim) - content transform
    b_c: jnp.ndarray             # (h_dim,)
    w_f: jnp.ndarray             # (input_dim,) - forget gate
    b_f: jnp.ndarray             # (1,)
    w_w: jnp.ndarray             # (input_dim,) - write gate
    b_w: jnp.ndarray             # (1,)
    w_r: jnp.ndarray             # (input_dim,) - read gate
    b_r: jnp.ndarray             # (1,)
    W_out: jnp.ndarray           # (output_dim, h_dim)


def gated_memory_step(params: GatedMemoryParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of gated working memory module.

    Args:
        params: GatedMemoryParams
        state: c - cell state (h_dim,)
        x: input (input_dim,)

    Returns:
        c_new: updated cell state (h_dim,)
        y: output (output_dim,)
        forget: forget gate value (scalar)
        write: write gate value (scalar)
        read: read gate value (scalar)
    """
    c = state

    # Gates
    forget = jax.nn.sigmoid(params.w_f @ x + params.b_f).squeeze()
    write = jax.nn.sigmoid(params.w_w @ x + params.b_w).squeeze()
    read = jax.nn.sigmoid(params.w_r @ x + params.b_r).squeeze()

    # Cell update
    content = jnp.tanh(params.W_c @ x + params.b_c)
    c_new = forget * c + write * content

    # Gated readout
    y = params.W_out @ (read * jnp.tanh(c_new))

    return c_new, y, forget, write, read


# =============================================================================
# Combined Modular Model
# =============================================================================

@struct.dataclass
class ModularParams:
    """Parameters for modular model with configurable modules per type.

    Can have different numbers of each module type (integrator, memory,
    oscillator, gated_memory).
    Selection via softmax(W_sel @ x + b_sel).
    Includes a "null" module that outputs zeros (allows skipping all modules).
    """
    integrators: Tuple[IntegratorParams, ...]  # K_int integrator modules
    memories: Tuple[MemoryParams, ...]  # K_mem memory modules
    oscillators: Tuple[OscillatorParams, ...]  # K_osc oscillator modules
    gated_memories: Tuple[GatedMemoryParams, ...]  # K_gm gated memory modules
    W_sel: jnp.ndarray  # (n_selections, input_dim) - selection weights (includes null)
    b_sel: jnp.ndarray  # (n_selections,) - selection biases (includes null)


def modular_forward(
    params: ModularParams,
    x_seq: jnp.ndarray,
    return_gates: bool = False,
    rng: Optional[jax.Array] = None,
    routing: Optional[jnp.ndarray] = None,
    temperature: float = 1.0,
):
    """Forward pass with modular selection.

    Args:
        params: ModularParams
        x_seq: Input sequence (seq_len, input_dim)
        return_gates: If True, return selection weights and internal gate values
        rng: Random key (currently unused)
        routing: Optional fixed routing vector (n_selections,). If provided, uses this
            instead of learned softmax selection. One-hot for hard routing:
            [1, 0, 0] = integrator, [0, 1, 0] = memory, [0, 0, 1] = null.
            Can also support soft mixtures: [0.5, 0.5, 0] = 50% int + 50% mem.

    Returns:
        If return_gates=False: ys (seq_len, output_dim)
        If return_gates=True: (ys, gates_dict) where gates_dict contains:
            - 'selections': (seq_len, total_modules) selection weights
            - 'integrator_{i}_alpha', 'integrator_{i}_beta': gates for i in 0..K_int-1
            - 'memory_{i}_omega': gates for i in 0..K_mem-1
    """
    del rng

    K_int = len(params.integrators)
    K_mem = len(params.memories)
    K_osc = len(params.oscillators)
    K_gm = len(params.gated_memories)
    total_modules = K_int + K_mem + K_osc + K_gm
    if total_modules == 0:
        raise ValueError("modular_forward requires at least one module")

    # Get dimensions from first module of each type (if exists)
    h_dim = params.integrators[0].W1.shape[0] if K_int > 0 else 0
    d_k_mem = params.memories[0].W_k.shape[0] if K_mem > 0 else 0
    d_v_mem = params.memories[0].W_v.shape[0] if K_mem > 0 else 0
    osc_dim = params.oscillators[0].W_enc.shape[0] if K_osc > 0 else 0
    gm_dim = params.gated_memories[0].W_c.shape[0] if K_gm > 0 else 0

    def step(states, x):
        int_states, mem_states, osc_states, gm_states = states

        all_outputs = []
        new_int_states = []
        new_mem_states = []
        new_osc_states = []
        new_gm_states = []

        # Collect gates if needed (will be concatenated into array)
        gate_values = [] if return_gates else None

        # Run all K_int integrator modules
        for i in range(K_int):
            h_new, y, alpha, beta = integrator_step(params.integrators[i], int_states[i], x)
            new_int_states.append(h_new)
            all_outputs.append(y)

            if return_gates:
                gate_values.extend([alpha, beta])

        # Run all K_mem memory modules
        for i in range(K_mem):
            S_new, y, omega = memory_step(params.memories[i], mem_states[i], x)
            new_mem_states.append(S_new)
            all_outputs.append(y)

            if return_gates:
                gate_values.append(omega)

        # Run all K_osc oscillator modules
        for i in range(K_osc):
            theta_new, y, gamma = oscillator_step(params.oscillators[i], osc_states[i], x)
            new_osc_states.append(theta_new)
            all_outputs.append(y)

            if return_gates:
                gate_values.append(gamma)

        # Run all K_gm gated memory modules
        for i in range(K_gm):
            c_new, y, f, w, r = gated_memory_step(params.gated_memories[i], gm_states[i], x)
            new_gm_states.append(c_new)
            all_outputs.append(y)

            if return_gates:
                gate_values.extend([f, w, r])

        # Stack outputs: (total_modules, output_dim)
        all_outputs = jnp.stack(all_outputs, axis=0)

        # Add null output (zeros) - allows model to "skip" all modules
        null_output = jnp.zeros(all_outputs.shape[1])  # (output_dim,)
        all_outputs = jnp.concatenate([all_outputs, null_output[None, :]], axis=0)

        # Selection over all modules + null
        if routing is not None:
            # Fixed routing: use provided weights directly
            sel_weights = routing  # (n_selections,)
        else:
            # Learned routing: softmax over logits (with temperature for annealing)
            sel_logits = params.W_sel @ x + params.b_sel  # (n_selections,)
            safe_temp = jnp.maximum(temperature, 1e-6)
            sel_weights = jax.nn.softmax(sel_logits / safe_temp)  # (n_selections,)

        # Soft selection (differentiable): y = sel_weights @ all_outputs
        y = sel_weights @ all_outputs  # (output_dim,)

        new_states = (new_int_states, new_mem_states, new_osc_states, new_gm_states)

        if return_gates:
            # Stack all gates into a single array for this timestep
            gates_array = jnp.array(gate_values)
            return new_states, (y, sel_weights, gates_array)
        else:
            return new_states, (y, sel_weights)

    # Initialize states for all modules
    int_states_0 = [jnp.zeros(h_dim) for _ in range(K_int)]
    mem_states_0 = [jnp.zeros((d_v_mem, d_k_mem)) for _ in range(K_mem)]
    osc_states_0 = [jnp.zeros(osc_dim) for _ in range(K_osc)]
    gm_states_0 = [jnp.zeros(gm_dim) for _ in range(K_gm)]
    initial_states = (int_states_0, mem_states_0, osc_states_0, gm_states_0)

    if return_gates:
        _, (ys, selections, all_gate_arrays) = jax.lax.scan(step, initial_states, x_seq)
        # Parse gate arrays into dict
        gates_dict = _parse_gate_arrays(all_gate_arrays, K_int, K_mem, K_osc, K_gm)
        gates_dict["out"] = selections
        return ys, gates_dict
    else:
        _, (ys, _selections) = jax.lax.scan(step, initial_states, x_seq)
        return ys


def _parse_gate_arrays(gate_arrays: jnp.ndarray, K_int: int, K_mem: int, K_osc: int = 0, K_gm: int = 0) -> dict:
    """Parse concatenated gate array into named dict.

    Args:
        gate_arrays: (seq_len, total_gates)
        K_int: Number of integrator modules
        K_mem: Number of memory modules
        K_osc: Number of oscillator modules
        K_gm: Number of gated memory modules

    Returns:
        Dict mapping gate names to (seq_len,) arrays
    """
    idx = 0
    gates = {}

    # Integrator gates: K_int * (alpha, beta)
    for i in range(K_int):
        gates[f'int{i}_alpha'] = gate_arrays[:, idx]
        idx += 1
        gates[f'int{i}_beta'] = gate_arrays[:, idx]
        idx += 1

    # Memory gates: K_mem * omega
    for i in range(K_mem):
        gates[f'mem{i}_omega'] = gate_arrays[:, idx]
        idx += 1

    # Oscillator gates: K_osc * gamma
    for i in range(K_osc):
        gates[f'osc{i}_gamma'] = gate_arrays[:, idx]
        idx += 1

    # Gated memory gates: K_gm * (forget, write, read)
    for i in range(K_gm):
        gates[f'gm{i}_forget'] = gate_arrays[:, idx]
        idx += 1
        gates[f'gm{i}_write'] = gate_arrays[:, idx]
        idx += 1
        gates[f'gm{i}_read'] = gate_arrays[:, idx]
        idx += 1

    return gates


@jax.jit
def modular_forward_jit(params: ModularParams, x_seq: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled forward pass (for training)."""
    return modular_forward(params, x_seq, return_gates=False)


# =============================================================================
# Initialization
# =============================================================================

def init_integrator_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
    num_freqs: int,
) -> IntegratorParams:
    """Initialize integrator module parameters."""
    keys = jax.random.split(rng, 7)
    scale = 0.1
    z_dim = 2 * h_dim * num_freqs

    return IntegratorParams(
        W1=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b1=jnp.zeros(h_dim),
        W2=jax.random.normal(keys[1], (h_dim, input_dim)) * scale,
        b2=jnp.zeros(h_dim),
        W_rec=jax.random.normal(keys[5], (h_dim, h_dim)) * 0.01,  # small init for stability
        w_alpha=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_alpha=jnp.zeros(1),
        w_beta=jax.random.normal(keys[3], (input_dim,)) * scale,
        b_beta=jnp.zeros(1),
        W_out=jax.random.normal(keys[4], (output_dim, z_dim)) * scale,
        freqs=2.0 ** jnp.arange(num_freqs),
    )


def init_memory_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    d_k: int,
    d_v: int,
) -> MemoryParams:
    """Initialize memory module parameters."""
    keys = jax.random.split(rng, 6)
    scale = 0.1

    return MemoryParams(
        W_k=jax.random.normal(keys[0], (d_k, input_dim)) * scale,
        W_v=jax.random.normal(keys[1], (d_v, input_dim)) * scale,
        W_q=jax.random.normal(keys[2], (d_k, input_dim)) * scale,
        w_omega=jax.random.normal(keys[3], (input_dim,)) * scale,
        b_omega=jnp.zeros(1),
        w_decay=jax.random.normal(keys[5], (input_dim,)) * scale,
        b_decay=jnp.ones(1) * 2.0,  # bias toward high decay (≈0.88) = strong retention
        W_out=jax.random.normal(keys[4], (output_dim, d_v)) * scale,
    )


def init_oscillator_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> OscillatorParams:
    """Initialize match/mismatch detector parameters."""
    keys = jax.random.split(rng, 5)
    scale = 0.1

    return OscillatorParams(
        W_enc=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b_enc=jnp.zeros(h_dim),
        W_probe=jax.random.normal(keys[1], (h_dim, input_dim)) * scale,
        b_probe=jnp.zeros(h_dim),
        w_gamma=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_gamma=jnp.zeros(1),
        W_out=jax.random.normal(keys[3], (output_dim, 2 * h_dim)) * scale,
    )


def init_gated_memory_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> GatedMemoryParams:
    """Initialize gated working memory module parameters."""
    keys = jax.random.split(rng, 5)
    scale = 0.1

    return GatedMemoryParams(
        W_c=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b_c=jnp.zeros(h_dim),
        w_f=jax.random.normal(keys[1], (input_dim,)) * scale,
        b_f=jnp.ones(1),  # bias toward remembering (forget gate starts high)
        w_w=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_w=jnp.zeros(1),
        w_r=jax.random.normal(keys[3], (input_dim,)) * scale,
        b_r=jnp.zeros(1),
        W_out=jax.random.normal(keys[4], (output_dim, h_dim)) * scale,
    )


def init_modular_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    K: Union[int, tuple] = 1,
    h_dim: int = 128,
    d_k: int = 128,
    d_v: int = 128,
    num_freqs: int = 4,
) -> ModularParams:
    """Initialize modular model parameters.

    Args:
        rng: Random key
        input_dim: Input dimension
        output_dim: Output dimension
        K: Number of modules per type. Can be:
            - int: same number for all module types
            - tuple of 2: (K_integrator, K_memory) - adds 1 osc + 1 gated_mem
            - tuple of 4: (K_int, K_mem, K_osc, K_gm)
        h_dim: Hidden dimension for integrators/oscillators/gated_memory
        d_k: Key dimension for memory
        d_v: Value dimension for memory
        num_freqs: Number of frequencies for positional encoding

    Returns:
        ModularParams with specified number of modules per type
    """
    # Parse K
    if isinstance(K, int):
        K_int, K_mem, K_osc, K_gm = K, K, 1, 1
    elif len(K) == 2:
        K_int, K_mem = K
        K_osc, K_gm = 1, 1  # always add 1 oscillator + 1 gated memory
    elif len(K) == 4:
        K_int, K_mem, K_osc, K_gm = K
    else:
        raise ValueError(f"K must be int or tuple of length 2 or 4, got: {K}")

    total_modules = K_int + K_mem + K_osc + K_gm
    n_selections = total_modules + 1  # +1 for null module

    # Split keys
    keys = jax.random.split(rng, total_modules + 1)
    key_idx = 0

    # Initialize integrator modules
    integrators = tuple(
        init_integrator_params(keys[key_idx + i], input_dim, output_dim, h_dim, num_freqs)
        for i in range(K_int)
    )
    key_idx += K_int

    # Initialize memory modules
    memories = tuple(
        init_memory_params(keys[key_idx + i], input_dim, output_dim, d_k, d_v)
        for i in range(K_mem)
    )
    key_idx += K_mem

    # Initialize oscillator modules
    oscillators = tuple(
        init_oscillator_params(keys[key_idx + i], input_dim, output_dim, h_dim)
        for i in range(K_osc)
    )
    key_idx += K_osc

    # Initialize gated memory modules
    gated_memories = tuple(
        init_gated_memory_params(keys[key_idx + i], input_dim, output_dim, h_dim)
        for i in range(K_gm)
    )

    # Selection weights - uniform init
    W_sel = jnp.zeros((n_selections, input_dim))
    b_sel = jnp.zeros(n_selections)

    return ModularParams(
        integrators=integrators,
        memories=memories,
        oscillators=oscillators,
        gated_memories=gated_memories,
        W_sel=W_sel,
        b_sel=b_sel,
    )


@struct.dataclass
class SequenceModelParams:
    """Wrapper params: embedding/projection + blocks + output."""

    embedding: Optional[jnp.ndarray] = None  # (vocab_size, embed_dim)
    W_in: Optional[jnp.ndarray] = None       # (embed_dim, input_dim)
    b_in: Optional[jnp.ndarray] = None       # (embed_dim,)

    block_params: tuple = None               # tuple of block params
    W_out: jnp.ndarray = None                # (output_dim, embed_dim)
    b_out: jnp.ndarray = None                # (output_dim,)

    W_cat: Optional[tuple] = None            # tuple of (embed_dim, 2*embed_dim)
    b_cat: Optional[tuple] = None            # tuple of (embed_dim,)


def make_sequence_model_forward(
    block_forward,
    n_layers,
    input_type="discrete",
    residual=False,
    routings=None,
    concat_input=False,
):
    """Create forward function for stacked blocks."""

    @jax.jit
    def forward(params: SequenceModelParams, x_seq: jnp.ndarray, rng: jax.Array, temperature: float = 1.0) -> jnp.ndarray:
        if input_type == "discrete":
            tokens = x_seq.squeeze().astype(jnp.int32)
            hidden = params.embedding[tokens]
        else:
            hidden = x_seq @ params.W_in.T + params.b_in

        input_embed = hidden
        n_blocks = len(params.block_params)
        rngs = jax.random.split(rng, n_layers)
        for layer_idx in range(n_layers):
            bp = params.block_params[layer_idx % n_blocks]
            layer_rng = rngs[layer_idx]
            routing = routings[layer_idx] if routings is not None else None
            block_out = block_forward(bp, hidden, layer_rng, routing=routing, temperature=temperature)
            if residual:
                block_out = hidden + block_out
            if concat_input:
                cat = jnp.concatenate([input_embed, block_out], axis=-1)
                hidden = cat @ params.W_cat[layer_idx % n_blocks].T + params.b_cat[layer_idx % n_blocks]
            else:
                hidden = block_out

        hidden = (hidden - jnp.mean(hidden, axis=-1, keepdims=True)) / (jnp.std(hidden, axis=-1, keepdims=True) + 1e-6)
        logits = hidden @ params.W_out.T + params.b_out
        return logits

    return forward


def make_sequence_model_forward_with_gates(
    block_forward_with_gates,
    n_layers,
    input_type="discrete",
    residual=False,
    routings=None,
    concat_input=False,
):
    """Create forward function that also returns gates."""

    def forward_with_gates(params: SequenceModelParams, x_seq: jnp.ndarray, temperature: float = 1.0):
        if input_type == "discrete":
            tokens = x_seq.squeeze().astype(jnp.int32)
            hidden = params.embedding[tokens]
        else:
            hidden = x_seq @ params.W_in.T + params.b_in

        input_embed = hidden
        n_blocks = len(params.block_params)
        all_layer_gates = {}
        for layer_idx in range(n_layers):
            bp = params.block_params[layer_idx % n_blocks]
            routing = routings[layer_idx] if routings is not None else None
            block_out, layer_gates = block_forward_with_gates(
                bp,
                hidden,
                return_gates=True,
                routing=routing,
                temperature=temperature,
            )
            if residual:
                block_out = hidden + block_out
            if concat_input:
                cat = jnp.concatenate([input_embed, block_out], axis=-1)
                hidden = cat @ params.W_cat[layer_idx % n_blocks].T + params.b_cat[layer_idx % n_blocks]
            else:
                hidden = block_out
            all_layer_gates[f"layer{layer_idx}"] = layer_gates

        hidden = (hidden - jnp.mean(hidden, axis=-1, keepdims=True)) / (jnp.std(hidden, axis=-1, keepdims=True) + 1e-6)
        logits = hidden @ params.W_out.T + params.b_out
        return logits, all_layer_gates

    return forward_with_gates


def make_loss_fn(forward_fn, label_smoothing: float = 0.0):
    """Create a jitted loss function for the given forward function."""

    @jax.jit
    def loss_fn(params, x_batch, y_batch, rng: jax.Array, temperature: float = 1.0):
        batch_size = x_batch.shape[0]
        rngs = jax.random.split(rng, batch_size)

        def single_loss(x, y, sample_rng):
            logits = forward_fn(params, x, sample_rng, temperature=temperature)
            mask = (y >= 0).astype(jnp.float32)
            safe_y = jnp.maximum(y, 0).astype(jnp.int32)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            if label_smoothing > 0:
                n_classes = logits.shape[-1]
                # Smooth target: (1-eps) on correct class + eps/n_classes uniform
                target_log_probs = jnp.take_along_axis(log_probs, safe_y[:, None], axis=-1).squeeze(-1)
                smooth_loss = -((1.0 - label_smoothing) * target_log_probs + label_smoothing * jnp.mean(log_probs, axis=-1))
                return jnp.sum(smooth_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            else:
                target_log_probs = jnp.take_along_axis(log_probs, safe_y[:, None], axis=-1).squeeze(-1)
                return -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        return jnp.mean(jax.vmap(single_loss)(x_batch, y_batch, rngs))

    return loss_fn


def make_sequence_model_spec(
    block_type,
    input_type="discrete",
    vocab_size=64,
    input_dim=None,
    output_dim=None,
    embed_dim=64,
    hidden_dim=128,
    d_k=64,
    d_v=64,
    n_layers=1,
    K=1,
    residual=False,
    routings=None,
    shared_weights=False,
    concat_input=False,
    label_smoothing=0.0,
):
    """Create ModelSpec for stacked modular blocks."""

    if block_type != "modular":
        raise ValueError(f"Only 'modular' block_type is supported, got: {block_type}")

    if output_dim is None:
        output_dim = vocab_size

    def block_forward(params, hidden, rng, routing=None, temperature=1.0):
        return modular_forward(params, hidden, rng=rng, routing=routing, temperature=temperature)

    def init_block(rng):
        return init_modular_params(
            rng,
            embed_dim,
            embed_dim,
            K=K,
            h_dim=hidden_dim,
            d_k=d_k,
            d_v=d_v,
        )

    def block_forward_with_gates(params, hidden, return_gates=False, routing=None, temperature=1.0):
        if return_gates:
            return modular_forward(params, hidden, return_gates=True, routing=routing, temperature=temperature)
        return modular_forward(params, hidden, return_gates=False, routing=routing, temperature=temperature)

    apply_with_gates = make_sequence_model_forward_with_gates(
        block_forward_with_gates,
        n_layers=n_layers,
        input_type=input_type,
        residual=residual,
        routings=routings,
        concat_input=concat_input,
    )

    # Number of parameterized blocks can be less than n_layers if shared weights
    num_blocks = 1 if shared_weights else n_layers

    def init(rng):
        keys = jax.random.split(rng, num_blocks + 3)
        if input_type == "discrete":
            embedding = jax.random.normal(keys[0], (vocab_size, embed_dim)) * 0.01
            W_in, b_in = None, None
        else:
            if input_dim is None:
                raise ValueError("input_dim required for continuous inputs")
            embedding = None
            W_in = jax.random.normal(keys[0], (embed_dim, input_dim)) * 0.1
            b_in = jnp.zeros(embed_dim)

        block_params = tuple(init_block(keys[i + 1]) for i in range(num_blocks))
        W_out = jax.random.normal(keys[-2], (output_dim, embed_dim)) * 0.1
        b_out = jnp.zeros(output_dim)

        if concat_input:
            W_cat = tuple(jax.random.normal(keys[i + 1], (embed_dim, 2 * embed_dim)) * 0.1 for i in range(num_blocks))
            b_cat = tuple(jnp.zeros(embed_dim) for _ in range(num_blocks))
        else:
            W_cat, b_cat = None, None

        return SequenceModelParams(
            embedding=embedding,
            W_in=W_in,
            b_in=b_in,
            block_params=block_params,
            W_out=W_out,
            b_out=b_out,
            W_cat=W_cat,
            b_cat=b_cat,
        )

    apply = make_sequence_model_forward(
        block_forward,
        n_layers=n_layers,
        input_type=input_type,
        residual=residual,
        routings=routings,
        concat_input=concat_input,
    )
    loss = make_loss_fn(apply, label_smoothing=label_smoothing)

    return ModelSpec(init=init, apply=apply, loss=loss, apply_with_gates=apply_with_gates)
