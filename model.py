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
    """Parameters for integrator module.
    
    h_{t+1} = (1-α_t) h_t + α_t (W_1 x_t + b_1) + β_t (W_2 x_t + b_2)
    z_t = [sin(h_t * freqs), cos(h_t * freqs)]
    y_t = W_out @ z_t
    """
    W1: jnp.ndarray              # (h_dim, input_dim)
    b1: jnp.ndarray              # (h_dim,)
    W2: jnp.ndarray              # (h_dim, input_dim)
    b2: jnp.ndarray              # (h_dim,)
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

    # Integrator update
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
    """Parameters for memory module (keys from x).
    
    k_t = normalize(W_k x_t)
    v_t = W_v x_t, q_t = normalize(W_q x_t)
    S_t = S_{t-1} - ω_t (S_{t-1} k_t - v_t) k_t^T
    m_t = S_t q_t
    y_t = W_out @ m_t
    """
    W_k: jnp.ndarray             # (d_k, input_dim)
    W_v: jnp.ndarray             # (d_v, input_dim)
    W_q: jnp.ndarray             # (d_k, input_dim)
    w_omega: jnp.ndarray         # (input_dim,)
    b_omega: jnp.ndarray         # (1,)
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

    # Delta rule update
    error = S @ k - v
    S_new = S - omega * jnp.outer(error, k)

    # Read
    m = S_new @ q

    # Output
    y = params.W_out @ m

    return S_new, y, omega


# =============================================================================
# Module 3: Sensory Gating (attention-inspired)
# =============================================================================

@struct.dataclass
class SensoryGateParams:
    """Parameters for sensory gating module.

    Inspired by top-down attention / gain modulation in sensory cortex.
    Computes a context-dependent gain vector that multiplicatively gates
    a learned transformation of the input. This helps the model focus
    on task-relevant input dimensions.

    g_t = sigmoid(W_g @ x_t + b_g)        -- gain vector (per-feature)
    z_t = g_t * (W_z @ x_t + b_z)         -- gated transform
    h_t = (1 - λ_t) * h_{t-1} + λ_t * z_t  -- smooth integration
    y_t = W_out @ h_t
    """
    W_g: jnp.ndarray         # (h_dim, input_dim) - gain computation
    b_g: jnp.ndarray         # (h_dim,)
    W_z: jnp.ndarray         # (h_dim, input_dim) - value transform
    b_z: jnp.ndarray         # (h_dim,)
    w_lam: jnp.ndarray       # (input_dim,) - integration rate
    b_lam: jnp.ndarray       # (1,)
    W_out: jnp.ndarray       # (output_dim, h_dim)


def sensory_gate_step(params: SensoryGateParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of sensory gating module.

    Args:
        params: SensoryGateParams
        state: h - hidden state (h_dim,)
        x: input (input_dim,)

    Returns:
        h_new: updated hidden state (h_dim,)
        y: output (output_dim,)
        lam: integration rate (scalar)
    """
    h = state

    # Gain vector: sigmoid gates per hidden dimension
    g = jax.nn.sigmoid(params.W_g @ x + params.b_g)

    # Value transform
    z = g * (params.W_z @ x + params.b_z)

    # Integration rate
    lam = jax.nn.sigmoid(params.w_lam @ x + params.b_lam).squeeze()

    # Smooth temporal integration
    h_new = (1 - lam) * h + lam * z

    # Output
    y = params.W_out @ h_new

    return h_new, y, lam


# =============================================================================
# Module 4: Lateral Inhibition (cortical competition)
# =============================================================================

@struct.dataclass
class LateralInhibitionParams:
    """Parameters for lateral inhibition module.

    Inspired by winner-take-all dynamics in cortical columns.
    Excitatory input drives hidden units, then lateral inhibition
    suppresses weakly active units, creating sparse task-selective
    representations.

    e_t = ReLU(W_e @ x_t + b_e)             -- excitation
    inh_t = W_inh @ e_t                      -- lateral inhibition (learned)
    a_t = ReLU(e_t - inh_t)                  -- post-inhibition activity (sparse)
    h_t = (1 - λ_t) * h_{t-1} + λ_t * a_t   -- temporal integration
    y_t = W_out @ h_t
    """
    W_e: jnp.ndarray          # (h_dim, input_dim) - excitatory weights
    b_e: jnp.ndarray          # (h_dim,)
    W_inh: jnp.ndarray        # (h_dim, h_dim) - lateral inhibition (off-diagonal)
    w_lam: jnp.ndarray        # (input_dim,) - integration rate
    b_lam: jnp.ndarray        # (1,)
    W_out: jnp.ndarray        # (output_dim, h_dim)


def lateral_inhibition_step(params: LateralInhibitionParams, state: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single step of lateral inhibition module."""
    h = state

    # Excitatory drive
    e = jax.nn.relu(params.W_e @ x + params.b_e)

    # Lateral inhibition: suppress weakly active units
    # Use abs to ensure inhibitory effect, zero diagonal to prevent self-inhibition
    W_inh_masked = params.W_inh * (1.0 - jnp.eye(params.W_inh.shape[0]))
    inh = jax.nn.relu(W_inh_masked) @ e  # positive inhibition only

    # Post-inhibition sparse activity
    a = jax.nn.relu(e - inh)

    # Integration rate
    lam = jax.nn.sigmoid(params.w_lam @ x + params.b_lam).squeeze()

    # Temporal smoothing
    h_new = (1 - lam) * h + lam * a

    # Output
    y = params.W_out @ h_new

    return h_new, y, lam


# =============================================================================
# Module 5: Comparator (match/mismatch detection)
# =============================================================================

@struct.dataclass
class ComparatorParams:
    """Parameters for comparator module.

    Inspired by prefrontal comparator circuits that detect match/mismatch
    between current input and a stored reference. Essential for
    match-to-sample, delayed comparison, and context-dependent tasks.

    ref_t = (1 - ω_t) * ref_{t-1} + ω_t * (W_ref @ x_t + b_ref)  -- update reference
    cur_t = W_cur @ x_t + b_cur                                     -- current encoding
    match_t = cur_t * ref_t                                          -- element-wise match
    diff_t = cur_t - ref_t                                           -- element-wise diff
    h_t = (1 - λ) * h_{t-1} + λ * [match; diff]                    -- integrate comparison
    y_t = W_out @ h_t
    """
    W_ref: jnp.ndarray       # (h_dim, input_dim) - reference encoding
    b_ref: jnp.ndarray       # (h_dim,)
    W_cur: jnp.ndarray       # (h_dim, input_dim) - current encoding
    b_cur: jnp.ndarray       # (h_dim,)
    w_omega: jnp.ndarray     # (input_dim,) - reference update gate
    b_omega: jnp.ndarray     # (1,)
    w_lam: jnp.ndarray       # (input_dim,) - integration rate
    b_lam: jnp.ndarray       # (1,)
    W_out: jnp.ndarray       # (output_dim, 2*h_dim) - reads from [match; diff]


def comparator_step(params: ComparatorParams, state, x: jnp.ndarray):
    """Single step of comparator module.

    State is (ref, h) where ref is the stored reference and h is the
    integrated comparison signal.
    """
    ref, h = state

    # Update reference (gated accumulation)
    omega = jax.nn.sigmoid(params.w_omega @ x + params.b_omega).squeeze()
    ref_input = params.W_ref @ x + params.b_ref
    ref_new = (1 - omega) * ref + omega * ref_input

    # Current encoding
    cur = params.W_cur @ x + params.b_cur

    # Comparison signals
    match = cur * ref_new           # element-wise similarity (high when aligned)
    diff = cur - ref_new            # element-wise difference (high when mismatched)
    comparison = jnp.concatenate([match, diff])  # (2*h_dim,)

    # Integration
    lam = jax.nn.sigmoid(params.w_lam @ x + params.b_lam).squeeze()
    h_new = (1 - lam) * h + lam * comparison

    # Output
    y = params.W_out @ h_new

    return (ref_new, h_new), y, omega


# =============================================================================
# Module 6: GRU (Gated Recurrent Unit)
# =============================================================================

@struct.dataclass
class GRUParams:
    """Parameters for GRU module.

    Inspired by ionic channel dynamics in neurons - update and reset
    gates control information flow analogous to voltage-gated channels.

    z_t = sigmoid(W_z @ [h_{t-1}, x_t])   -- update gate
    r_t = sigmoid(W_r @ [h_{t-1}, x_t])   -- reset gate
    h_hat = tanh(W_h @ [r_t * h_{t-1}, x_t]) -- candidate
    h_t = (1 - z_t) * h_{t-1} + z_t * h_hat  -- update
    y_t = W_out @ h_t
    """
    W_z: jnp.ndarray      # (h_dim, h_dim + input_dim) - update gate
    b_z: jnp.ndarray      # (h_dim,)
    W_r: jnp.ndarray      # (h_dim, h_dim + input_dim) - reset gate
    b_r: jnp.ndarray      # (h_dim,)
    W_h: jnp.ndarray      # (h_dim, h_dim + input_dim) - candidate
    b_h: jnp.ndarray      # (h_dim,)
    W_out: jnp.ndarray    # (output_dim, h_dim)


def gru_step(params: GRUParams, state: jnp.ndarray, x: jnp.ndarray):
    """Single step of GRU module."""
    h = state
    hx = jnp.concatenate([h, x])

    z = jax.nn.sigmoid(params.W_z @ hx + params.b_z)   # update gate
    r = jax.nn.sigmoid(params.W_r @ hx + params.b_r)   # reset gate

    rhx = jnp.concatenate([r * h, x])
    h_hat = jnp.tanh(params.W_h @ rhx + params.b_h)    # candidate

    h_new = (1 - z) * h + z * h_hat                     # update
    y = params.W_out @ h_new

    return h_new, y, jnp.mean(z)  # return mean update gate as diagnostic


# =============================================================================
# Module 8: LSTM (Long Short-Term Memory)
# =============================================================================

@struct.dataclass
class LSTMParams:
    W_f: jnp.ndarray      # (h_dim, h_dim + input_dim) - forget gate
    b_f: jnp.ndarray      # (h_dim,)
    W_i: jnp.ndarray      # (h_dim, h_dim + input_dim) - input gate
    b_i: jnp.ndarray      # (h_dim,)
    W_c: jnp.ndarray      # (h_dim, h_dim + input_dim) - candidate
    b_c: jnp.ndarray      # (h_dim,)
    W_o: jnp.ndarray      # (h_dim, h_dim + input_dim) - output gate
    b_o: jnp.ndarray      # (h_dim,)
    W_out: jnp.ndarray    # (output_dim, h_dim)


def lstm_step(params: LSTMParams, state, x: jnp.ndarray):
    h, c = state
    hx = jnp.concatenate([h, x])
    f = jax.nn.sigmoid(params.W_f @ hx + params.b_f)
    i = jax.nn.sigmoid(params.W_i @ hx + params.b_i)
    c_hat = jnp.tanh(params.W_c @ hx + params.b_c)
    c_new = f * c + i * c_hat
    o = jax.nn.sigmoid(params.W_o @ hx + params.b_o)
    h_new = o * jnp.tanh(c_new)
    y = params.W_out @ h_new
    return (h_new, c_new), y, jnp.mean(f)


def init_lstm_params(rng, input_dim, output_dim, h_dim):
    keys = jax.random.split(rng, 5)
    concat_dim = h_dim + input_dim
    scale = (2.0 / concat_dim) ** 0.5
    return LSTMParams(
        W_f=jax.random.normal(keys[0], (h_dim, concat_dim)) * scale,
        b_f=jnp.ones(h_dim),  # forget bias=1: keep memories by default
        W_i=jax.random.normal(keys[1], (h_dim, concat_dim)) * scale,
        b_i=jnp.zeros(h_dim),
        W_c=jax.random.normal(keys[2], (h_dim, concat_dim)) * scale,
        b_c=jnp.zeros(h_dim),
        W_o=jax.random.normal(keys[3], (h_dim, concat_dim)) * scale,
        b_o=jnp.zeros(h_dim),
        W_out=jax.random.normal(keys[4], (output_dim, h_dim)) * (2.0 / (output_dim + h_dim)) ** 0.5,
    )


# =============================================================================
# Module 7: Reservoir / Echo State Network
# =============================================================================

@struct.dataclass
class ReservoirParams:
    """Parameters for reservoir (echo state network) module.

    Inspired by cortical microcircuits with rich recurrent dynamics.
    The recurrent weights W_rec are FIXED (not trained) - initialized
    at the spectral radius to ensure echo state property. Only the
    input projection W_in_res and readout W_out are trained.

    h_t = tanh(W_rec @ h_{t-1} + W_in_res @ x_t + b_in)  -- reservoir dynamics
    y_t = W_out @ h_t
    """
    W_rec: jnp.ndarray       # (h_dim, h_dim) - FIXED recurrent weights
    W_in_res: jnp.ndarray    # (h_dim, input_dim) - input projection (trained)
    b_in: jnp.ndarray        # (h_dim,)
    W_out: jnp.ndarray       # (output_dim, h_dim) - readout (trained)
    leak_rate: jnp.ndarray   # (1,) - leaky integration rate


def reservoir_step(params: ReservoirParams, state: jnp.ndarray, x: jnp.ndarray):
    """Single step of reservoir module."""
    h = state
    leak = jax.nn.sigmoid(params.leak_rate).squeeze()
    # Stop gradient on W_rec: recurrent weights are fixed (echo state property)
    W_rec_fixed = jax.lax.stop_gradient(params.W_rec)
    pre = W_rec_fixed @ h + params.W_in_res @ x + params.b_in
    h_new = (1 - leak) * h + leak * jnp.tanh(pre)
    y = params.W_out @ h_new
    return h_new, y, leak


# =============================================================================
# Combined Modular Model
# =============================================================================

@struct.dataclass
class ModularParams:
    """Parameters for modular model with configurable modules per type.

    Module types: integrator, memory, sensory_gate, lateral_inhibition.
    Selection via softmax(W_sel @ x + b_sel).
    Includes a "null" module that outputs zeros (allows skipping all modules).
    Neuromodulatory gain: per-module scalar that scales output (like dopamine/NE).
    """
    integrators: Tuple[IntegratorParams, ...]
    memories: Tuple[MemoryParams, ...]
    sensory_gates: Tuple[SensoryGateParams, ...]
    lateral_inhibitions: Tuple[LateralInhibitionParams, ...]
    comparators: Tuple[ComparatorParams, ...]
    grus: Tuple[GRUParams, ...]
    lstms: Tuple[LSTMParams, ...]
    reservoirs: Tuple[ReservoirParams, ...]
    W_sel: jnp.ndarray  # (n_selections, input_dim)
    b_sel: jnp.ndarray  # (n_selections,)
    W_gain: jnp.ndarray  # (total_modules, input_dim) - neuromodulatory gain per module
    b_gain: jnp.ndarray  # (total_modules,)
    log_sigma: jnp.ndarray  # (1,) - learnable divisive normalization sigma
    W_sigma: jnp.ndarray    # (1, input_dim) - input-dependent sigma modulation
    b_sigma: jnp.ndarray    # (1,)


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
    K_sg = len(params.sensory_gates)
    K_li = len(params.lateral_inhibitions)
    K_cmp = len(params.comparators)
    K_gru = len(params.grus)
    K_lstm = len(params.lstms)
    K_res = len(params.reservoirs)
    total_modules = K_int + K_mem + K_sg + K_li + K_cmp + K_gru + K_lstm + K_res
    if total_modules == 0:
        raise ValueError("modular_forward requires at least one module")

    # Get dimensions from first module of each type (if exists)
    h_dim = params.integrators[0].W1.shape[0] if K_int > 0 else 0
    d_k_mem = params.memories[0].W_k.shape[0] if K_mem > 0 else 0
    d_v_mem = params.memories[0].W_v.shape[0] if K_mem > 0 else 0
    sg_dim = params.sensory_gates[0].W_g.shape[0] if K_sg > 0 else 0
    li_dim = params.lateral_inhibitions[0].W_e.shape[0] if K_li > 0 else 0
    cmp_dim = params.comparators[0].W_ref.shape[0] if K_cmp > 0 else 0
    gru_dim = params.grus[0].W_z.shape[0] if K_gru > 0 else 0
    lstm_dim = params.lstms[0].W_f.shape[0] if K_lstm > 0 else 0
    res_dim = params.reservoirs[0].W_rec.shape[0] if K_res > 0 else 0

    def step(states, x):
        int_states, mem_states, sg_states, li_states, cmp_states, gru_states, lstm_states, res_states = states

        all_outputs = []
        new_int_states = []
        new_mem_states = []
        new_sg_states = []
        new_li_states = []
        new_cmp_states = []
        new_gru_states = []
        new_lstm_states = []
        new_res_states = []

        gate_values = [] if return_gates else None

        for i in range(K_int):
            h_new, y, alpha, beta = integrator_step(params.integrators[i], int_states[i], x)
            new_int_states.append(h_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.extend([alpha, beta])

        for i in range(K_mem):
            S_new, y, omega = memory_step(params.memories[i], mem_states[i], x)
            new_mem_states.append(S_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(omega)

        for i in range(K_sg):
            h_new, y, lam = sensory_gate_step(params.sensory_gates[i], sg_states[i], x)
            new_sg_states.append(h_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(lam)

        for i in range(K_li):
            h_new, y, lam = lateral_inhibition_step(params.lateral_inhibitions[i], li_states[i], x)
            new_li_states.append(h_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(lam)

        for i in range(K_cmp):
            state_new, y, omega = comparator_step(params.comparators[i], cmp_states[i], x)
            new_cmp_states.append(state_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(omega)

        for i in range(K_gru):
            h_new, y, z_mean = gru_step(params.grus[i], gru_states[i], x)
            new_gru_states.append(h_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(z_mean)

        for i in range(K_lstm):
            state_new, y, f_mean = lstm_step(params.lstms[i], lstm_states[i], x)
            new_lstm_states.append(state_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(f_mean)

        for i in range(K_res):
            h_new, y, leak = reservoir_step(params.reservoirs[i], res_states[i], x)
            new_res_states.append(h_new)
            all_outputs.append(y)
            if return_gates:
                gate_values.append(leak)

        # Stack outputs: (total_modules, output_dim) where total_modules = K_int + K_mem
        all_outputs = jnp.stack(all_outputs, axis=0)

        # Neuromodulatory gain: context-dependent scaling per module
        # Inspired by dopamine/norepinephrine modulation of cortical circuits
        gain = jax.nn.sigmoid(params.W_gain @ x + params.b_gain)  # (total_modules,)
        all_outputs = all_outputs * gain[:, None]  # scale each module output

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

        # Select output (weighted sum with softmax, but since it's selection, use argmax-like behavior)
        # For hard selection: y = all_outputs[argmax(sel_weights)]
        # For soft selection (differentiable): y = sel_weights @ all_outputs
        y = sel_weights @ all_outputs  # (output_dim,)

        # Divisive normalization: canonical neural computation for contrast invariance
        # y / (sigma + ||y||) where sigma adapts to input context
        base_sigma = jnp.exp(params.log_sigma).squeeze()
        sigma_mod = jax.nn.softplus(params.W_sigma @ x + params.b_sigma).squeeze()
        sigma = base_sigma + sigma_mod  # input-adaptive sigma
        y = y / (sigma + jnp.sqrt(jnp.sum(y ** 2) + 1e-8))

        new_states = (new_int_states, new_mem_states, new_sg_states, new_li_states, new_cmp_states, new_gru_states, new_lstm_states, new_res_states)

        if return_gates:
            # Stack all gates into a single array for this timestep
            gates_array = jnp.array(gate_values)
            return new_states, (y, sel_weights, gates_array)
        else:
            return new_states, (y, sel_weights)
    
    # Initialize states for all modules
    int_states_0 = [jnp.zeros(h_dim) for _ in range(K_int)]
    mem_states_0 = [jnp.zeros((d_v_mem, d_k_mem)) for _ in range(K_mem)]
    sg_states_0 = [jnp.zeros(sg_dim) for _ in range(K_sg)]
    li_states_0 = [jnp.zeros(li_dim) for _ in range(K_li)]
    # Comparator state: (ref, h) where ref is (h_dim,) and h is (2*h_dim,)
    cmp_states_0 = [(jnp.zeros(cmp_dim), jnp.zeros(2 * cmp_dim)) for _ in range(K_cmp)]
    gru_states_0 = [jnp.zeros(gru_dim) for _ in range(K_gru)]
    lstm_states_0 = [(jnp.zeros(lstm_dim), jnp.zeros(lstm_dim)) for _ in range(K_lstm)]
    res_states_0 = [jnp.zeros(res_dim) for _ in range(K_res)]
    initial_states = (int_states_0, mem_states_0, sg_states_0, li_states_0, cmp_states_0, gru_states_0, lstm_states_0, res_states_0)

    if return_gates:
        _, (ys, selections, all_gate_arrays) = jax.lax.scan(step, initial_states, x_seq)
        # Parse gate arrays into dict
        gates_dict = _parse_gate_arrays(all_gate_arrays, K_int, K_mem, K_sg, K_li, K_cmp, K_gru, K_lstm, K_res)
        gates_dict["out"] = selections
        return ys, gates_dict
    else:
        _, (ys, _selections) = jax.lax.scan(step, initial_states, x_seq)
        return ys


def _parse_gate_arrays(gate_arrays: jnp.ndarray, K_int: int, K_mem: int, K_sg: int = 0, K_li: int = 0, K_cmp: int = 0, K_gru: int = 0, K_lstm: int = 0, K_res: int = 0) -> dict:
    """Parse concatenated gate array into named dict."""
    idx = 0
    gates = {}

    for i in range(K_int):
        gates[f'int{i}_alpha'] = gate_arrays[:, idx]
        idx += 1
        gates[f'int{i}_beta'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_mem):
        gates[f'mem{i}_omega'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_sg):
        gates[f'sg{i}_lambda'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_li):
        gates[f'li{i}_lambda'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_cmp):
        gates[f'cmp{i}_omega'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_gru):
        gates[f'gru{i}_z'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_lstm):
        gates[f'lstm{i}_f'] = gate_arrays[:, idx]
        idx += 1

    for i in range(K_res):
        gates[f'res{i}_leak'] = gate_arrays[:, idx]
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
    keys = jax.random.split(rng, 6)
    scale = 0.1
    z_dim = 2 * h_dim * num_freqs
    
    return IntegratorParams(
        W1=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b1=jnp.zeros(h_dim),
        W2=jax.random.normal(keys[1], (h_dim, input_dim)) * scale,
        b2=jnp.zeros(h_dim),
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
    keys = jax.random.split(rng, 5)
    scale = 0.1
    
    return MemoryParams(
        W_k=jax.random.normal(keys[0], (d_k, input_dim)) * scale,
        W_v=jax.random.normal(keys[1], (d_v, input_dim)) * scale,
        W_q=jax.random.normal(keys[2], (d_k, input_dim)) * scale,
        w_omega=jax.random.normal(keys[3], (input_dim,)) * scale,
        b_omega=jnp.zeros(1),
        W_out=jax.random.normal(keys[4], (output_dim, d_v)) * scale,
    )


def init_sensory_gate_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> SensoryGateParams:
    """Initialize sensory gating module parameters."""
    keys = jax.random.split(rng, 4)
    scale = 0.1
    return SensoryGateParams(
        W_g=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b_g=jnp.zeros(h_dim),
        W_z=jax.random.normal(keys[1], (h_dim, input_dim)) * scale,
        b_z=jnp.zeros(h_dim),
        w_lam=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_lam=jnp.zeros(1),
        W_out=jax.random.normal(keys[3], (output_dim, h_dim)) * scale,
    )


def init_lateral_inhibition_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> LateralInhibitionParams:
    """Initialize lateral inhibition module parameters."""
    keys = jax.random.split(rng, 4)
    scale = 0.1
    return LateralInhibitionParams(
        W_e=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b_e=jnp.zeros(h_dim),
        W_inh=jax.random.normal(keys[1], (h_dim, h_dim)) * (scale * 0.5),
        w_lam=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_lam=jnp.zeros(1),
        W_out=jax.random.normal(keys[3], (output_dim, h_dim)) * scale,
    )


def init_gru_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> GRUParams:
    """Initialize GRU module parameters."""
    keys = jax.random.split(rng, 4)
    scale = 0.1
    cat_dim = h_dim + input_dim
    return GRUParams(
        W_z=jax.random.normal(keys[0], (h_dim, cat_dim)) * scale,
        b_z=jnp.zeros(h_dim),
        W_r=jax.random.normal(keys[1], (h_dim, cat_dim)) * scale,
        b_r=jnp.zeros(h_dim),
        W_h=jax.random.normal(keys[2], (h_dim, cat_dim)) * scale,
        b_h=jnp.zeros(h_dim),
        W_out=jax.random.normal(keys[3], (output_dim, h_dim)) * scale,
    )


def init_reservoir_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
    spectral_radius: float = 0.9,
) -> ReservoirParams:
    """Initialize reservoir module parameters.

    W_rec is initialized as a sparse random matrix scaled to the desired
    spectral radius. This is NOT trained - only W_in_res and W_out are.
    """
    keys = jax.random.split(rng, 4)
    scale = 0.1

    # Create random recurrent matrix and scale to spectral radius
    W_rec = jax.random.normal(keys[0], (h_dim, h_dim)) / jnp.sqrt(h_dim)
    # Make it sparse: zero out ~80% of connections
    mask = jax.random.bernoulli(keys[1], p=0.2, shape=(h_dim, h_dim))
    W_rec = W_rec * mask
    # Scale to desired spectral radius (approximate)
    W_rec = W_rec * spectral_radius

    return ReservoirParams(
        W_rec=W_rec,
        W_in_res=jax.random.normal(keys[2], (h_dim, input_dim)) * scale,
        b_in=jnp.zeros(h_dim),
        W_out=jax.random.normal(keys[3], (output_dim, h_dim)) * scale,
        leak_rate=jnp.zeros(1),  # sigmoid(0) = 0.5
    )


def init_comparator_params(
    rng: jax.Array,
    input_dim: int,
    output_dim: int,
    h_dim: int,
) -> ComparatorParams:
    """Initialize comparator module parameters."""
    keys = jax.random.split(rng, 5)
    scale = 0.1
    return ComparatorParams(
        W_ref=jax.random.normal(keys[0], (h_dim, input_dim)) * scale,
        b_ref=jnp.zeros(h_dim),
        W_cur=jax.random.normal(keys[1], (h_dim, input_dim)) * scale,
        b_cur=jnp.zeros(h_dim),
        w_omega=jax.random.normal(keys[2], (input_dim,)) * scale,
        b_omega=jnp.zeros(1),
        w_lam=jax.random.normal(keys[3], (input_dim,)) * scale,
        b_lam=jnp.zeros(1),
        W_out=jax.random.normal(keys[4], (output_dim, 2 * h_dim)) * scale,
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
            - int: same number for integrators and memories
            - tuple: (K_integrator, K_memory)
        h_dim: Hidden dimension for integrators
        d_k: Key dimension for memory
        d_v: Value dimension for memory
        num_freqs: Number of frequencies for positional encoding

    Returns:
        ModularParams with specified number of modules per type
    """
    # Parse K
    if isinstance(K, int):
        K_int, K_mem = K, K
    else:
        if len(K) != 2:
            raise ValueError(f"K must be int or tuple of length 2, got: {K}")
        K_int, K_mem = K

    K_sg = 1  # Always add 1 sensory gating module
    K_li = 1  # Always add 1 lateral inhibition module
    K_cmp = 1  # Always add 1 comparator module
    K_gru = 1  # Always add 1 GRU module
    K_lstm = 1  # Always add 1 LSTM module
    K_res = 1  # Always add 1 reservoir module
    total_modules = K_int + K_mem + K_sg + K_li + K_cmp + K_gru + K_lstm + K_res
    n_selections = total_modules + 1  # +1 for null module

    keys = jax.random.split(rng, total_modules + 1)

    integrators = tuple(
        init_integrator_params(keys[i], input_dim, output_dim, h_dim, num_freqs)
        for i in range(K_int)
    )

    memories = tuple(
        init_memory_params(keys[K_int + i], input_dim, output_dim, d_k, d_v)
        for i in range(K_mem)
    )

    sensory_gates = tuple(
        init_sensory_gate_params(keys[K_int + K_mem + i], input_dim, output_dim, h_dim)
        for i in range(K_sg)
    )

    lateral_inhibitions = tuple(
        init_lateral_inhibition_params(keys[K_int + K_mem + K_sg + i], input_dim, output_dim, h_dim)
        for i in range(K_li)
    )

    comparators = tuple(
        init_comparator_params(keys[K_int + K_mem + K_sg + K_li + i], input_dim, output_dim, h_dim)
        for i in range(K_cmp)
    )

    grus = tuple(
        init_gru_params(keys[K_int + K_mem + K_sg + K_li + K_cmp + i], input_dim, output_dim, h_dim)
        for i in range(K_gru)
    )

    lstms = tuple(
        init_lstm_params(keys[K_int + K_mem + K_sg + K_li + K_cmp + K_gru + i], input_dim, output_dim, h_dim)
        for i in range(K_lstm)
    )

    reservoirs = tuple(
        init_reservoir_params(keys[K_int + K_mem + K_sg + K_li + K_cmp + K_gru + K_lstm + i], input_dim, output_dim, h_dim)
        for i in range(K_res)
    )

    W_sel = jnp.zeros((n_selections, input_dim))
    b_sel = jnp.zeros(n_selections)

    # Neuromodulatory gain: initialized to bias=2 so sigmoid(2)≈0.88 (near 1, minimal effect initially)
    W_gain = jnp.zeros((total_modules, input_dim))
    b_gain = jnp.full(total_modules, 2.0)

    # Learnable sigma for divisive normalization (init to log(1)=0)
    log_sigma = jnp.zeros(1)
    W_sigma = jnp.zeros((1, input_dim))
    b_sigma = jnp.zeros(1)

    return ModularParams(
        integrators=integrators,
        memories=memories,
        sensory_gates=sensory_gates,
        lateral_inhibitions=lateral_inhibitions,
        comparators=comparators,
        grus=grus,
        lstms=lstms,
        reservoirs=reservoirs,
        W_sel=W_sel,
        b_sel=b_sel,
        W_gain=W_gain,
        b_gain=b_gain,
        log_sigma=log_sigma,
        W_sigma=W_sigma,
        b_sigma=b_sigma,
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

        logits = hidden @ params.W_out.T + params.b_out
        return logits, all_layer_gates

    return forward_with_gates


def make_loss_fn(forward_fn):
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
            n_classes = logits.shape[-1]
            # Label smoothing: mix one-hot with uniform
            smooth = 0.1
            one_hot = jax.nn.one_hot(safe_y, n_classes)
            soft_targets = (1.0 - smooth) * one_hot + smooth / n_classes
            per_step_loss = -jnp.sum(soft_targets * log_probs, axis=-1)
            # L2 penalty on logits to prevent them from growing too large
            logit_penalty = 5e-4 * jnp.mean(logits ** 2, axis=-1)
            per_step_loss = per_step_loss + logit_penalty
            return jnp.sum(per_step_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

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
    loss = make_loss_fn(apply)

    return ModelSpec(init=init, apply=apply, loss=loss, apply_with_gates=apply_with_gates)
