import jax
import jax.numpy as jnp


def _append_zero_feature(X):
    """Append an always-zero feature channel so routers can select a null input."""

    zero_col = jnp.zeros(X.shape[:-1] + (1,), dtype=X.dtype)
    return jnp.concatenate([X, zero_col], axis=-1)


def make_dataset(rng, name, T, dt, n_samples, targets=None, delay_durations=None):
    N = T // dt
    fix_duration = 1500
    stim_onset = 1000
    stim_duration = 200
    omega = 0.01

    if targets is None:
        n_targets = 5
        targets = jnp.arange(n_targets) + 1
    if delay_durations is None:
        delay_durations = jnp.arange(500, 1000, 100)

    X = []
    Y = []
    metadata = []

    if name == "delay_go":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.choice(_rng, targets)
            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[(stim_onset//dt):(stim_onset//dt + stim_duration//dt), 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(stim_value)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'stimulus_onset_ms': stim_onset,
                'stimulus_offset_ms': stim_onset + stim_duration,
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "delay_anti_go":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.choice(_rng, targets)
            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[(stim_onset//dt):(stim_onset//dt + stim_duration//dt), 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(len(targets) - stim_value + 1)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'stimulus_onset_ms': stim_onset,
                'stimulus_offset_ms': stim_onset + stim_duration,
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "integration":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            delay_duration = jax.random.choice(_rng, delay_durations)
            stim_onset = fix_duration - delay_duration

            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[(stim_onset//dt) - ((stim_duration//dt)//2):(stim_onset//dt) + ((stim_duration//dt)//2), 1].set(1)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(delay_duration * omega)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'delay_duration_ms': float(delay_duration),
                'target_value': float(delay_duration * omega),
                'stimulus_onset_ms': float(stim_onset - stim_duration//2),
                'stimulus_offset_ms': float(stim_onset + stim_duration//2),
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "sequence":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.choice(_rng, targets)

            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[:fix_duration//dt, 1].set(stim_value)

            response_duration = T - fix_duration
            # ramp from stim_value to stim_value * response_duration * omega
            ramp = jnp.linspace(stim_value, stim_value + response_duration * omega, response_duration//dt)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:fix_duration//dt + response_duration//dt].set(ramp)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'ramp_start_value': float(stim_value),
                'ramp_end_value': float(stim_value + response_duration * omega),
                'ramp_duration_ms': response_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    X = jnp.stack(X)  # (n_samples, T, n_features)
    X = _append_zero_feature(X)
    Y = jnp.stack(Y)  # (n_samples, T)

    return X, Y, metadata


def make_dataset_continous(rng, name, T, dt, n_samples, target_range=None, delay_duration_range=None, stim_onset=None):
    N = T // dt
    fix_duration = 1500
    if stim_onset is None:
        stim_onset = 1000
    stim_duration = 200
    omega = 0.01

    if target_range is None:
        target_range = (1.0, 6.0)
    if delay_duration_range is None:
        delay_duration_range = (500.0, 1000.0)

    X = []
    Y = []
    metadata = []

    if name == "delay_go":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.uniform(_rng, minval=target_range[0], maxval=target_range[1])
            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[(stim_onset//dt):(stim_onset//dt + stim_duration//dt), 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(stim_value)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'stimulus_onset_ms': stim_onset,
                'stimulus_offset_ms': stim_onset + stim_duration,
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "delay_anti_go":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.uniform(_rng, minval=target_range[0], maxval=target_range[1])
            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[(stim_onset//dt):(stim_onset//dt + stim_duration//dt), 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(target_range[0] + target_range[1] - stim_value)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'stimulus_onset_ms': stim_onset,
                'stimulus_offset_ms': stim_onset + stim_duration,
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "integration":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            delay_duration = jax.random.uniform(_rng, minval=delay_duration_range[0], maxval=delay_duration_range[1])
            stim_onset = fix_duration - delay_duration
            stim_onset_idx = int(stim_onset // dt)

            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[stim_onset_idx - ((stim_duration//dt)//2):stim_onset_idx + ((stim_duration//dt)//2), 1].set(1)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:].set(delay_duration * omega)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'delay_duration_ms': float(delay_duration),
                'target_value': float(delay_duration * omega),
                'stimulus_onset_ms': float(stim_onset - stim_duration//2),
                'stimulus_offset_ms': float(stim_onset + stim_duration//2),
                'stimulus_duration_ms': stim_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "sequence":
        for i in range(n_samples):
            _rng, rng = jax.random.split(rng)
            stim_value = jax.random.uniform(_rng, minval=target_range[0], maxval=target_range[1])

            x = jnp.zeros((N, 2))
            x = x.at[:fix_duration//dt, 0].set(1)
            x = x.at[:fix_duration//dt, 1].set(stim_value)

            response_duration = T - fix_duration
            ramp = jnp.linspace(stim_value, stim_value + response_duration * omega, response_duration//dt)

            y = jnp.zeros((N,))
            y = y.at[fix_duration//dt:fix_duration//dt + response_duration//dt].set(ramp)

            X.append(x)
            Y.append(y)

            metadata.append({
                'trial_id': i,
                'dataset_name': name,
                'stimulus_value': float(stim_value),
                'ramp_start_value': float(stim_value),
                'ramp_end_value': float(stim_value + response_duration * omega),
                'ramp_duration_ms': response_duration,
                'fix_duration_ms': fix_duration,
                'response_onset_ms': fix_duration
            })

    elif name == "associative_recall":
        # Associative recall task following zoology structure (vectorized)
        # Input: token indices [k1, v1, k2, v2, ..., q1, q2, ..., qN]
        # Output: -1 during storage, value token during query
        import numpy as np

        num_kv_pairs = 4
        vocab_size = 64
        key_vocab_size = vocab_size // 2
        context_size = num_kv_pairs * 2
        seq_len = context_size + num_kv_pairs

        # Use numpy for fast generation (then convert to jax)
        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        # Sample keys and values for all samples at once
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys = np.array([np_rng.choice(key_choices, size=num_kv_pairs, replace=False)
                         for _ in range(n_samples)])  # (n_samples, num_kv_pairs)
        values = np.array([np_rng.choice(value_choices, size=num_kv_pairs, replace=False)
                          for _ in range(n_samples)])  # (n_samples, num_kv_pairs)
        query_orders = np.array([np_rng.permutation(num_kv_pairs)
                                 for _ in range(n_samples)])  # (n_samples, num_kv_pairs)

        # Build sequences
        X = np.zeros((n_samples, N), dtype=np.float32)
        Y = np.full((n_samples, N), -1, dtype=np.float32)

        # Storage phase: interleave keys and values
        X[:, 0::2][:, :num_kv_pairs] = keys
        X[:, 1::2][:, :num_kv_pairs] = values

        # Query phase
        for i in range(n_samples):
            for j in range(num_kv_pairs):
                query_idx = context_size + j
                orig_idx = query_orders[i, j]
                X[i, query_idx] = keys[i, orig_idx]
                Y[i, query_idx] = values[i, orig_idx]

        # Fill padding with random tokens
        if N > seq_len:
            X[:, seq_len:] = np_rng.integers(0, vocab_size, size=(n_samples, N - seq_len))

        # Convert to jax arrays
        X = jnp.array(X)
        Y = jnp.array(Y)

        metadata = [{'trial_id': i, 'dataset_name': name, 'num_kv_pairs': num_kv_pairs,
                     'vocab_size': vocab_size, 'seq_len': seq_len} for i in range(n_samples)]

        return X, Y, metadata

    elif name == "integration_discrete":
        # Integration task in discrete token format
        # Model must output delay token continuously during response period
        # Input:  [PAD, ..., STIM, STIM, ..., STIM, PAD, ..., GO, PAD, ..., PAD]
        # Target: [-1,  ..., -1,   -1,   ..., -1,   -1,  ..., DELAY_k, ..., DELAY_k]
        #                                                     ^-- response starts at GO
        import numpy as np

        PAD, STIM, GO = 0, 1, 2
        DELAY_START = 3
        seq_len = N

        # Timing parameters (matching original continuous task)
        # Original: fix_duration=1500ms, stim_onset=variable, stim_duration=200ms, T=2000ms, dt=20ms
        # With seq_len=100: fix_idx=75, stim_duration=10
        stim_duration = 10
        fix_idx = int(seq_len * 0.75)  # GO signal and response starts at 75% of sequence

        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        X = np.zeros((n_samples, seq_len), dtype=np.float32)
        Y = np.full((n_samples, seq_len), -1, dtype=np.float32)

        # Get allowed delays from delay_duration_range
        # Delay = timesteps from stim center to fix_idx (response onset)
        if delay_duration_range is not None:
            min_delay, max_delay = int(delay_duration_range[0]), int(delay_duration_range[1])
        else:
            min_delay = seq_len // 4
            max_delay = fix_idx - stim_duration

        # Number of unique delay values determines output vocab
        num_delay_bins = max_delay - min_delay + 1

        # Sample delays for all samples
        allowed_delays = np.arange(min_delay, max_delay + 1)
        delays = np_rng.choice(allowed_delays, size=n_samples)

        for i in range(n_samples):
            delay = delays[i]
            # Stimulus center position (delay measured from center to response onset)
            stim_center = fix_idx - delay
            stim_start = max(0, stim_center - stim_duration // 2)
            stim_end = min(fix_idx, stim_start + stim_duration)

            # Map delay to output token (delay -> bin index -> token)
            delay_bin = delay - min_delay  # 0-indexed bin
            target_token = DELAY_START + delay_bin

            # Set stimulus tokens
            X[i, stim_start:stim_end] = STIM

            # Set GO signal during entire response period
            X[i, fix_idx:] = GO

            # Set target: output delay token continuously during response period
            Y[i, fix_idx:] = target_token

        X = jnp.array(X)
        Y = jnp.array(Y)

        vocab_size = DELAY_START + num_delay_bins
        metadata = [{'trial_id': i, 'dataset_name': name, 'vocab_size': vocab_size,
                     'num_delay_bins': num_delay_bins, 'seq_len': seq_len,
                     'stim_duration': stim_duration, 'fix_idx': fix_idx,
                     'delay': int(delays[i]), 'min_delay': min_delay, 'max_delay': max_delay,
                     'stim_center': int(fix_idx - delays[i])} for i in range(n_samples)]

        return X, Y, metadata

    elif name == "path_integration":
        # Path Integration task - track state through action sequence
        # Predict state when revisiting (no observations, just state tracking)
        # Input:  [action1, action2, action3, ...]
        # Target: [-1,      state2,  -1,      ...]  (predict on revisit)
        import numpy as np

        # Config
        num_attributes = 2
        num_values_per_attribute = 4
        num_states = num_values_per_attribute ** num_attributes  # 16 states
        seq_len = N

        # Vocab: actions [0, num_actions), states for output [num_actions, num_actions + num_states)
        num_actions = 16
        STATE_OFFSET = num_actions  # output state tokens start here
        vocab_size = num_actions + num_states

        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        # Build action effects (deterministic per action, same across all sequences)
        action_effects = np.zeros((num_actions, 2), dtype=int)
        effect_rng = np.random.default_rng(42)  # Fixed seed for consistent action semantics
        action_effects[:, 0] = effect_rng.integers(0, num_attributes, size=num_actions)
        action_effects[:, 1] = effect_rng.integers(1, num_values_per_attribute, size=num_actions)

        X = np.zeros((n_samples, seq_len), dtype=np.float32)
        Y = np.full((n_samples, seq_len), -1, dtype=np.float32)

        dims = (num_values_per_attribute,) * num_attributes

        for i in range(n_samples):
            # Random initial state
            curr_state = np_rng.integers(0, num_values_per_attribute, size=num_attributes)
            visited_states = set()
            initial_flat = np.ravel_multi_index(curr_state, dims)
            visited_states.add(initial_flat)

            # Apply random actions
            for step in range(seq_len):
                action_idx = np_rng.integers(0, num_actions)
                action_token = action_idx

                # Apply action
                attr_idx, shift = action_effects[action_idx]
                curr_state = curr_state.copy()
                curr_state[attr_idx] = (curr_state[attr_idx] + shift) % num_values_per_attribute
                curr_flat = np.ravel_multi_index(curr_state, dims)

                # Set input
                X[i, step] = action_token

                # Target: predict state if revisiting
                if curr_flat in visited_states:
                    Y[i, step] = STATE_OFFSET + curr_flat
                else:
                    visited_states.add(curr_flat)

        X = jnp.array(X)
        Y = jnp.array(Y)

        metadata = [{'trial_id': i, 'dataset_name': name, 'vocab_size': vocab_size,
                     'num_states': num_states, 'num_actions': num_actions,
                     'num_attributes': num_attributes, 'num_values_per_attribute': num_values_per_attribute,
                     'seq_len': seq_len} for i in range(n_samples)]

        return X, Y, metadata

    elif name == "path_integration_4actions":
        # Path Integration with 4 cardinal actions (up, right, down, left)
        # Cleaner task structure with interpretable actions
        # Input:  [action1, action2, action3, ...]
        # Target: [-1,      state2,  -1,      ...]  (predict on revisit)
        import numpy as np

        # Config
        num_values_per_attribute = 4
        num_states = num_values_per_attribute ** 2  # 4x4 = 16 states
        seq_len = N

        # 4 actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
        # Token layout: actions [0-3], positions [4-19], stimuli [20-35]
        num_actions = 4
        num_positions = num_states  # 16 positions
        num_stimuli = 16  # observation tokens (for navigation task compatibility)
        POS_OFFSET = num_actions
        STIM_OFFSET = POS_OFFSET + num_positions
        vocab_size = num_actions + num_positions + num_stimuli

        # Action effects: (row_delta, col_delta)
        # UP: row -1 (mod 4) = +3
        # RIGHT: col +1
        # DOWN: row +1
        # LEFT: col -1 (mod 4) = +3
        action_effects = np.array([
            [3, 0],  # UP: row += 3 (mod 4) = row -= 1
            [0, 1],  # RIGHT: col += 1
            [1, 0],  # DOWN: row += 1
            [0, 3],  # LEFT: col += 3 (mod 4) = col -= 1
        ], dtype=int)

        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        # seq_len = 1 (init state) + N-1 (actions)
        X = np.zeros((n_samples, seq_len), dtype=np.float32)
        Y = np.full((n_samples, seq_len), -1, dtype=np.float32)

        for i in range(n_samples):
            # Random initial state
            row = np_rng.integers(0, num_values_per_attribute)
            col = np_rng.integers(0, num_values_per_attribute)
            init_state = row * num_values_per_attribute + col
            visited_states = set()
            visited_states.add(init_state)

            # First token is initial position
            X[i, 0] = POS_OFFSET + init_state

            for step in range(1, seq_len):
                action = np_rng.integers(0, num_actions)

                # Apply action
                row = (row + action_effects[action, 0]) % num_values_per_attribute
                col = (col + action_effects[action, 1]) % num_values_per_attribute
                state_flat = row * num_values_per_attribute + col

                X[i, step] = action

                # Target: predict position if revisiting
                if state_flat in visited_states:
                    Y[i, step] = POS_OFFSET + state_flat
                else:
                    visited_states.add(state_flat)

        X = jnp.array(X)
        Y = jnp.array(Y)

        metadata = [{'trial_id': i, 'dataset_name': name, 'vocab_size': vocab_size,
                     'num_states': num_states, 'num_actions': num_actions,
                     'seq_len': seq_len, 'actions': ['UP', 'RIGHT', 'DOWN', 'LEFT']}
                    for i in range(n_samples)]

        return X, Y, metadata

    elif name == "navigation":
        # Navigation / Action-State Associative Recall task
        # Agent navigates on state grid, must recall observations for revisited states
        # Input:  [action1, obs1, action2, obs2, ...]
        # Labels: [-1,      -1,   -1,      obs2, ...]  (unmasked only for revisits)
        import numpy as np

        # Config
        num_attributes = 2
        num_values_per_attribute = 4
        num_states = num_values_per_attribute ** num_attributes  # 16 states
        seq_len = N  # Use N from dt/T calculation

        # Vocab split: keys [1, vocab_size//2), values [vocab_size//2, vocab_size)
        vocab_size = 64
        key_vocab_size = vocab_size // 2

        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        # Build key effects (deterministic per key, same across all sequences)
        # Each key modifies exactly one attribute by a fixed shift
        key_effects = np.zeros((key_vocab_size, 2), dtype=int)
        effect_rng = np.random.default_rng(42)  # Fixed seed for consistent key semantics
        key_effects[:, 0] = effect_rng.integers(0, num_attributes, size=key_vocab_size)
        key_effects[:, 1] = effect_rng.integers(1, num_values_per_attribute, size=key_vocab_size)

        X = np.zeros((n_samples, seq_len), dtype=np.float32)
        Y = np.full((n_samples, seq_len), -1, dtype=np.float32)  # -1 for masked

        dims = (num_values_per_attribute,) * num_attributes
        num_steps = seq_len // 2
        key_range = np.arange(1, key_vocab_size)
        value_range = np.arange(key_vocab_size, vocab_size)

        for i in range(n_samples):
            # Random observation mapping for this sequence
            obs_tokens = np_rng.choice(value_range, size=num_states, replace=False)
            visited_states = set()
            curr_state = np_rng.integers(0, num_values_per_attribute, size=num_attributes)

            for step in range(num_steps):
                action_token = np_rng.choice(key_range)
                attr_idx, shift = key_effects[action_token - 1]
                next_state = curr_state.copy()
                next_state[attr_idx] = (next_state[attr_idx] + shift) % num_values_per_attribute
                flat_idx = np.ravel_multi_index(next_state, dims)
                obs_token = obs_tokens[flat_idx]

                # Positions in sequence
                action_pos = step * 2
                obs_pos = step * 2 + 1

                # Set input tokens
                X[i, action_pos] = action_token
                X[i, obs_pos] = obs_token

                # Label: unmasked only for revisits
                if flat_idx in visited_states:
                    Y[i, action_pos] = obs_token  # Predict obs after action for revisited state
                else:
                    visited_states.add(flat_idx)
                # obs position is always masked (model doesn't predict after seeing obs)

                curr_state = next_state

        X = jnp.array(X)
        Y = jnp.array(Y)

        metadata = [{'trial_id': i, 'dataset_name': name, 'vocab_size': vocab_size,
                     'num_attributes': num_attributes, 'num_values_per_attribute': num_values_per_attribute,
                     'num_states': num_states, 'seq_len': seq_len} for i in range(n_samples)]

        return X, Y, metadata

    elif name == "navigation_4actions":
        # Navigation with 4 cardinal actions (combines path_integration_4actions + associative_recall)
        # Agent navigates on 4x4 grid with UP/RIGHT/DOWN/LEFT
        # Each state has a random observation (fixed per sequence)
        # On revisit, predict the observation
        # Input:  [action, obs, action, obs, ...]
        # Labels: [-1,     -1,  -1,     obs, ...]  (unmasked only at action pos for revisits)
        import numpy as np

        # Config
        num_values_per_attribute = 4
        num_states = num_values_per_attribute ** 2  # 4x4 = 16 states
        num_actions = 4  # UP, RIGHT, DOWN, LEFT
        num_positions = num_states  # 16 positions
        num_stimuli = 16  # stimulus/observation tokens

        # Token layout: actions [0-3], positions [4-19], stimuli [20-35]
        POS_OFFSET = num_actions
        STIM_OFFSET = POS_OFFSET + num_positions
        vocab_size = num_actions + num_positions + num_stimuli
        num_steps = N // 2  # Each step = action + obs
        seq_len = num_steps * 2

        # Action effects: (row_delta, col_delta) with wraparound
        action_effects = np.array([
            [3, 0],  # UP: row -= 1 (mod 4)
            [0, 1],  # RIGHT: col += 1
            [1, 0],  # DOWN: row += 1
            [0, 3],  # LEFT: col -= 1 (mod 4)
        ], dtype=int)

        seed = rng  # rng is already an int seed
        np_rng = np.random.default_rng(seed)

        # seq_len = 1 (init state) + (num_steps-1) * 2 (action, obs pairs) + 1 padding
        # Keep seq_len = num_steps * 2 for consistency
        X = np.zeros((n_samples, seq_len), dtype=np.float32)
        Y = np.full((n_samples, seq_len), -1, dtype=np.float32)
        init_states = []

        for i in range(n_samples):
            # Random stimulus per state (fixed for this sequence)
            obs_tokens = np_rng.permutation(num_stimuli) + STIM_OFFSET

            # Random initial state
            row = np_rng.integers(0, num_values_per_attribute)
            col = np_rng.integers(0, num_values_per_attribute)
            init_state = row * num_values_per_attribute + col
            init_states.append(init_state)
            visited_states = {init_state: obs_tokens[init_state]}

            # First token is initial observation
            X[i, 0] = obs_tokens[init_state]

            for step in range(num_steps - 1):
                action = np_rng.integers(0, num_actions)

                # Apply action
                row = (row + action_effects[action, 0]) % num_values_per_attribute
                col = (col + action_effects[action, 1]) % num_values_per_attribute
                state_flat = row * num_values_per_attribute + col
                obs_token = obs_tokens[state_flat]

                action_pos = 1 + step * 2
                obs_pos = 1 + step * 2 + 1

                X[i, action_pos] = action
                X[i, obs_pos] = obs_token

                # Predict observation on revisit (at action position, before seeing obs)
                if state_flat in visited_states:
                    Y[i, action_pos] = obs_token
                else:
                    visited_states[state_flat] = obs_token

        X = jnp.array(X)
        Y = jnp.array(Y)

        # Store init_states for visualization
        metadata = [{'trial_id': i, 'dataset_name': name, 'vocab_size': vocab_size,
                     'num_states': num_states, 'num_actions': num_actions,
                     'seq_len': seq_len, 'actions': ['UP', 'RIGHT', 'DOWN', 'LEFT'],
                     'init_state': init_states[i]}
                    for i in range(n_samples)]

        return X, Y, metadata

    X = jnp.stack(X)  # (n_samples, T, n_features)
    X = _append_zero_feature(X)
    Y = jnp.stack(Y)  # (n_samples, T)

    return X, Y, metadata


def make_multitask_dataset_continuous(rng, task_names, T, dt, n_samples, target_range=None, delay_duration_range=None):
    """Generate multi-task dataset with task one-hot encoding added to input."""
    N = T // dt
    fix_duration = 1500
    stim_onset = 1000
    stim_duration = 200
    omega = 0.01

    if target_range is None:
        target_range = (1.0, 6.0)
    if delay_duration_range is None:
        delay_duration_range = (500.0, 1000.0)

    num_tasks = len(task_names)
    X = []
    Y = []
    metadata = []

    for i in range(n_samples):
        rng, task_rng = jax.random.split(rng)
        task_idx = jax.random.choice(task_rng, num_tasks)
        task_name = task_names[int(task_idx)]

        task_onehot = jnp.zeros(num_tasks)
        task_onehot = task_onehot.at[task_idx].set(1.0)

        if task_name == "delay_go":
            rng, stim_rng = jax.random.split(rng)
            stim_value = jax.random.uniform(stim_rng, minval=target_range[0], maxval=target_range[1])
            x = jnp.zeros((N, 2))
            fix_idx = fix_duration // dt
            stim_start_idx = stim_onset // dt
            stim_end_idx = stim_start_idx + stim_duration // dt
            x = x.at[:fix_idx, 0].set(1)
            x = x.at[stim_start_idx:stim_end_idx, 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_idx:].set(stim_value)

            metadata.append({'trial_id': i, 'task_name': task_name, 'task_idx': int(task_idx)})

        elif task_name == "delay_anti_go":
            rng, stim_rng = jax.random.split(rng)
            stim_value = jax.random.uniform(stim_rng, minval=target_range[0], maxval=target_range[1])
            x = jnp.zeros((N, 2))
            fix_idx = fix_duration // dt
            stim_start_idx = stim_onset // dt
            stim_end_idx = stim_start_idx + stim_duration // dt
            x = x.at[:fix_idx, 0].set(1)
            x = x.at[stim_start_idx:stim_end_idx, 1].set(stim_value)

            y = jnp.zeros((N,))
            y = y.at[fix_idx:].set(target_range[0] + target_range[1] - stim_value)

            metadata.append({'trial_id': i, 'task_name': task_name, 'task_idx': int(task_idx)})

        elif task_name == "integration":
            rng, delay_rng = jax.random.split(rng)
            delay_duration = jax.random.uniform(delay_rng, minval=delay_duration_range[0], maxval=delay_duration_range[1])
            stim_onset_val = fix_duration - delay_duration
            stim_onset_idx = int(stim_onset_val // dt)
            stim_half_dur = (stim_duration // dt) // 2
            fix_idx = fix_duration // dt

            x = jnp.zeros((N, 2))
            x = x.at[:fix_idx, 0].set(1)
            x = x.at[stim_onset_idx - stim_half_dur:stim_onset_idx + stim_half_dur, 1].set(1)

            y = jnp.zeros((N,))
            y = y.at[fix_idx:].set(delay_duration * omega)

            metadata.append({'trial_id': i, 'task_name': task_name, 'task_idx': int(task_idx)})

        elif task_name == "sequence":
            rng, stim_rng = jax.random.split(rng)
            stim_value = jax.random.uniform(stim_rng, minval=target_range[0], maxval=target_range[1])

            x = jnp.zeros((N, 2))
            fix_idx = fix_duration // dt
            x = x.at[:fix_idx, 0].set(1)
            x = x.at[:fix_idx, 1].set(stim_value)

            response_duration = T - fix_duration
            response_dur_idx = response_duration // dt
            ramp = jnp.linspace(stim_value, stim_value + response_duration * omega, response_dur_idx)

            y = jnp.zeros((N,))
            y = y.at[fix_idx:fix_idx + response_dur_idx].set(ramp)

            metadata.append({'trial_id': i, 'task_name': task_name, 'task_idx': int(task_idx)})

        task_onehot_expanded = jnp.tile(task_onehot, (N, 1))
        x_with_task = jnp.concatenate([x, task_onehot_expanded], axis=1)

        X.append(x_with_task)
        Y.append(y)

    X = jnp.stack(X)
    X = _append_zero_feature(X)
    Y = jnp.stack(Y)

    return X, Y, metadata


def make_neurogym_dataset(seed, task_factory, task_cfg, n_samples, condition_filter=None):
    """Generate dataset from NeuroGym task.

    Args:
        seed: Random seed (int)
        task_factory: Factory function from extended_yang19 (e.g., go, anti)
        task_cfg: NeurogymTaskConfig with dim_ring, dt, seq_len, input_dim
        n_samples: Number of samples to generate
        condition_filter: Optional dict specifying which conditions to keep, e.g.:
            {'delay': (0, 800)} - keep delays in range [0, 800]
            {'delay': [0, 100, 200]} - keep only these specific delays
            {'ground_truth': [0, 1, 2, 3]} - keep only these directions

    Returns:
        X: (n_samples, seq_len, input_dim) - numpy array
        Y: (n_samples, seq_len) - numpy array
        metadata: list of trial info dicts
    """
    import numpy as np

    env = task_factory(dt=task_cfg.dt, dim_ring=task_cfg.dim_ring)

    X = []
    Y = []
    metadata = []

    np_rng = np.random.default_rng(seed)

    # Keep generating until we have enough valid samples
    max_attempts = n_samples * 10
    attempts = 0

    while len(X) < n_samples and attempts < max_attempts:
        attempts += 1
        env.seed(int(np_rng.integers(0, 2**31)))
        env.reset()
        trial_info = env.new_trial()

        # Check condition filter
        if condition_filter is not None:
            keep = True

            # Filter by delay
            if 'delay' in condition_filter:
                delay = env.timing['delay']
                allowed = condition_filter['delay']
                if isinstance(allowed, tuple) and len(allowed) == 2:
                    keep = keep and (allowed[0] <= delay <= allowed[1])
                elif isinstance(allowed, (list, np.ndarray)):
                    keep = keep and (delay in allowed)

            # Filter by ground_truth direction
            if 'ground_truth' in condition_filter:
                gt = trial_info['ground_truth']
                allowed = condition_filter['ground_truth']
                if isinstance(allowed, (list, np.ndarray)):
                    keep = keep and (gt in allowed)
                elif isinstance(allowed, tuple) and len(allowed) == 2:
                    keep = keep and (allowed[0] <= gt <= allowed[1])

            if not keep:
                continue

        # Get observations and ground truth from env
        ob = env.unwrapped.ob
        gt = env.unwrapped.gt

        trial_len = min(len(ob), task_cfg.seq_len)

        # Pad to seq_len
        x = np.zeros((task_cfg.seq_len, task_cfg.input_dim), dtype=np.float32)
        y = np.zeros((task_cfg.seq_len,), dtype=np.int32)
        x[:trial_len] = ob[:trial_len]
        y[:trial_len] = gt[:trial_len]  # 0=fixation, 1-16=choice

        X.append(x)
        Y.append(y)
        metadata.append({
            'trial_id': len(X) - 1,
            'task_name': task_factory.__name__ if hasattr(task_factory, '__name__') else str(task_factory),
            'trial_info': trial_info,
            'trial_len': trial_len,
        })

    if len(X) < n_samples:
        raise RuntimeError(
            f"Could not generate enough samples: got {len(X)}/{n_samples} after {max_attempts} attempts. "
            f"Condition filter may be too restrictive: {condition_filter}"
        )

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y, metadata
