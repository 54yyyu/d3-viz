Visualization Data Format Documentation

  Overview

  The visualization feature saves intermediate data during D3-DNA diffusion sampling for analysis and visualization.
  This data captures the complete diffusion process including sequences, score matrices, noise levels, and optional
  oracle predictions at each sampling step.

  Data Structure

  File Formats

  - HDF5 (.h5, .hdf5) - Recommended for hierarchical data organization
  - NPZ (.npz) - Compressed NumPy format for simpler access

  Data Organization

  visualization_data.h5/npz
  ├── metadata/
  │   ├── dataset: "deepstarr"
  │   ├── num_samples: 100
  │   ├── sequence_length: 249
  │   ├── total_steps: 249
  │   ├── architecture: "transformer"
  │   ├── split: "test" (evaluation only)
  │   ├── save_oracle_mse: true/false
  │   ├── original_samples: (batch_size, seq_length) [evaluation only]
  │   ├── ground_truth_labels: (batch_size, num_labels) [evaluation only]
  │   ├── ground_truth_predictions: (batch_size, num_labels) [evaluation only]
  │   ├── dataset_indices: (batch_size,)
  │   └── noise_schedule/
  │       ├── type: "geometric"
  │       ├── sigma_min: 0.001
  │       └── sigma_max: 1.0
  └── steps/
      ├── step_0000/
      │   ├── step: 0
      │   ├── timestep: 1.0
      │   ├── noise_level: 1.0
      │   ├── noise_rate: 0.1
      │   ├── sequence: (batch_size, seq_length)
      │   ├── score_matrix: (batch_size, seq_length, 4)
      │   ├── prob_matrix: (batch_size, seq_length, 4)
      │   ├── oracle_mse: (batch_size,) [evaluation only]
      │   └── oracle_predictions: (batch_size, num_labels) [evaluation only]
      ├── step_0001/
      │   └── ...
      └── step_0248/
          └── ... (final denoising step)

  Data Arrays

  Per-Step Data

  | Field        | Shape                       | Type       | Description                                        |
  |--------------|-----------------------------|------------|----------------------------------------------------|
  | sequence     | (batch_size, seq_length)    | int64      | Token indices (0=A, 1=C, 2=G, 3=T) at current step |
  | score_matrix | (batch_size, seq_length, 4) | float16/32 | Model's score predictions for each nucleotide      |
  | prob_matrix  | (batch_size, seq_length, 4) | float16/32 | Probability matrix from staggered score computation |
  | attribution_matrix | (batch_size, seq_length, 4) | float16/32 | GradientSHAP attribution scores for each nucleotide [analysis only] |
  | noise_level  | scalar                      | float32    | Current noise level (σ)                            |
  | noise_rate   | scalar                      | float32    | Rate of noise change (dσ/dt)                       |
  | oracle_mse        | (batch_size,)               | float32    | Oracle MSE vs ground truth [evaluation only]      |
  | oracle_predictions| (batch_size, num_labels)   | float32    | Oracle predictions at current step [evaluation only]|

  Metadata

  | Field           | Type   | Description                                           |
  |-----------------|--------|-------------------------------------------------------|
  | dataset          | string                   | Dataset name (e.g., "deepstarr", "mpra")              |
  | num_samples      | int                      | Number of sequences sampled                           |
  | sequence_length  | int                      | Length of each sequence                               |
  | total_steps      | int                      | Total sampling steps                                  |
  | architecture     | string                   | Model architecture ("transformer", "convolutional")   |
  | split            | string                   | Dataset split for evaluation ("train", "val", "test") |
  | save_oracle_mse  | bool                     | Whether oracle MSE data is included                   |
  | original_samples | (batch_size, seq_length) | Original test samples for MSE comparison [evaluation only] |
  | ground_truth_labels    | (batch_size, num_labels) | Ground truth target labels [evaluation only]         |
  | ground_truth_predictions| (batch_size, num_labels)| Oracle predictions on ground truth data [evaluation only]|
  | dataset_indices        | (batch_size,)            | Dataset indices used for sample selection            |

  Loading Data

  Python (HDF5)

  import h5py
  import numpy as np

  # Load visualization data
  with h5py.File('visualization_data.h5', 'r') as f:
      # Read metadata
      dataset_name = f['metadata'].attrs['dataset']
      num_samples = f['metadata'].attrs['num_samples']
      total_steps = f['metadata'].attrs['total_steps']

      # Read step data
      step_0 = f['steps/step_0000']
      sequences = np.array(step_0['sequence'])      # (batch_size, seq_length)
      scores = np.array(step_0['score_matrix'])     # (batch_size, seq_length, 4)
      probs = np.array(step_0['prob_matrix'])       # (batch_size, seq_length, 4)
      noise_level = step_0.attrs['noise_level']    # scalar

      # Oracle data (if available)
      if 'oracle_mse' in step_0:
          oracle_mse = np.array(step_0['oracle_mse'])  # (batch_size,)
      if 'oracle_predictions' in step_0:
          oracle_predictions = np.array(step_0['oracle_predictions'])  # (batch_size, num_labels)
      
      # Ground truth data (if available, stored in metadata)
      if 'original_samples' in f['metadata']:
          original_samples = np.array(f['metadata']['original_samples'])  # (batch_size, seq_length)
      if 'ground_truth_labels' in f['metadata']:
          gt_labels = np.array(f['metadata']['ground_truth_labels'])  # (batch_size, num_labels)
      if 'ground_truth_predictions' in f['metadata']:
          gt_predictions = np.array(f['metadata']['ground_truth_predictions'])  # (batch_size, num_labels)
      if 'dataset_indices' in f['metadata']:
          dataset_indices = np.array(f['metadata']['dataset_indices'])  # (batch_size,)

  Python (NPZ)

  import numpy as np

  # Load visualization data
  data = np.load('visualization_data.npz')

  # Read metadata
  dataset_name = str(data['meta_dataset'])
  num_samples = int(data['meta_num_samples'])
  total_steps = int(data['meta_total_steps'])

  # Read step data (stacked arrays)
  sequences = data['sequences']        # (steps, batch_size, seq_length)
  score_matrices = data['score_matrices']  # (steps, batch_size, seq_length, 4)
  prob_matrices = data['prob_matrices']    # (steps, batch_size, seq_length, 4)
  noise_levels = data['noise_levels']      # (steps,)
  timesteps = data['timesteps']            # (steps,)

  # Oracle data (if available)
  if 'oracle_mses' in data:
      oracle_mses = data['oracle_mses']    # (steps, batch_size)
  if 'oracle_predictions' in data:
      oracle_predictions = data['oracle_predictions']  # (steps, batch_size, num_labels)
  
  # Ground truth data (if available)
  if 'original_samples' in data:
      original_samples = data['original_samples']  # (batch_size, seq_length)
  if 'ground_truth_labels' in data:
      gt_labels = data['ground_truth_labels']  # (batch_size, num_labels)
  if 'ground_truth_predictions' in data:
      gt_predictions = data['ground_truth_predictions']  # (batch_size, num_labels)
  if 'dataset_indices' in data:
      dataset_indices = data['dataset_indices']  # (batch_size,)

  Dataset Index Selection

  Sample Selection Options

  - **Random sampling**: `--max_samples 100` (default behavior)
  - **Specific indices**: `--specific_indices "11,12,40"` (guarantees these samples)
  - **Hybrid selection**: Both arguments together for guaranteed samples + random fill

  The `dataset_indices` metadata tracks which samples were actually selected, ensuring
  reproducibility and enabling targeted analysis of specific sequences.

  Sampling vs Evaluation Data

  Sampling Mode

  - Oracle MSE: Not included (save_oracle_mse: false)
  - Split: Not applicable
  - Use case: Understanding unconditional/conditional generation process

  Evaluation Mode

  - Oracle MSE: Included (save_oracle_mse: true)
  - Ground truth data: Included (labels, predictions, original samples)
  - Oracle predictions: Included at each step for trend analysis
  - Split: Specified (typically "test")
  - Use case: Analyzing generation quality and comparing to ground truth

  Memory Usage

  | Samples | Steps | Seq Length | Approx Size (HDF5) |
  |---------|-------|------------|--------------------|
  | 32      | 249   | 249        | ~45 MB             |
  | 100     | 249   | 249        | ~140 MB            |
  | 1000    | 249   | 249        | ~1.4 GB            |

  Note: Size includes sequences (int64), score/prob matrices (float16), and metadata

  Example Usage Commands

  Generate Sampling Visualization

  python model_zoo/deepstarr/sample.py \
      --checkpoint path/to/checkpoint.ckpt \
      --architecture transformer \
      --config model_zoo/deepstarr/configs/transformer.yaml \
      --num_samples 100 \
      --save_viz_data \
      --viz_format hdf5 \
      --viz_output my_sampling_viz.h5

  Generate Evaluation Visualization

  python model_zoo/deepstarr/evaluate.py \
      --checkpoint path/to/checkpoint.ckpt \
      --oracle_checkpoint path/to/oracle.ckpt \
      --data_path model_zoo/deepstarr/DeepSTARR_data.h5 \
      --save_viz_data \
      --viz_output my_evaluation_viz.h5 \
      --max_samples 500 \
      --specific_indices "11,12,40"  # Optional: guarantee specific samples

  # Select specific samples with random fill
  python model_zoo/lentimpra/evaluate.py \
      --checkpoint path/to/checkpoint.ckpt \
      --oracle_checkpoint path/to/oracle.ckpt \
      --data_path path/to/lentimpra_data.h5 \
      --save_viz_data \
      --max_samples 100 \
      --specific_indices "5,15,25,35"  # These 4 guaranteed + 96 random
