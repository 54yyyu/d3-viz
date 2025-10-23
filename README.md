# D3-DNA Diffusion Visualizer

An interactive web-based visualization tool for analyzing and exploring D3-DNA diffusion model outputs. This tool provides comprehensive visualization of DNA sequence generation processes, including score matrices, probability distributions, attribution maps, and motif analysis.

## Overview

This visualizer is designed to analyze intermediate data from D3-DNA diffusion sampling processes. It enables researchers to:

- Visualize DNA sequence generation across diffusion timesteps
- Analyze score matrices and probability distributions at each step
- Explore attribution maps using GradientSHAP
- Discover and compare motifs using FIMO and TOMTOM algorithms
- Track oracle predictions and ground truth comparisons
- Perform variant effect prediction (VEP) analysis

## Features

### Core Visualization Components

- **Sequence Viewer**: Interactive DNA sequence visualization with synchronized scrolling
- **Score Matrix Heatmaps**: Visualize model score predictions for each nucleotide position
- **Probability Matrix Visualization**: Display probability distributions across positions
- **Attribution Maps**: Analyze feature importance using GradientSHAP scores
- **Sequence Logos**: Generate and display Position Weight Matrices (PWMs) using LogoJS
- **Timeline Slider**: Navigate through diffusion sampling steps
- **Oracle Predictions**: Track prediction quality during generation (evaluation mode)
- **Variant Effect Prediction**: Analyze sequence variants and their predicted effects

### Motif Analysis Tools

- **FIMO Integration**: Scan sequences for known transcription factor binding sites
- **TOMTOM Analysis**: Compare discovered motifs against JASPAR database
- **TF-MoDISco**: Discover motifs from attribution scores
- **JASPAR Database**: Access to 2024 CORE non-redundant motifs

### Visualization Modes

- **Both View**: Side-by-side comparison of two samples or timesteps
- **Heatmap Mode**: Score/probability matrix visualization
- **Logo Mode**: Sequence logo representations
- **Attribution Mode**: Feature importance visualization

### UI Features

- **Dark/Light Theme**: Toggle between visual themes
- **Synchronized Scrolling**: Coordinate multiple visualization panels
- **Responsive Design**: Adapts to different screen sizes
- **Export Functionality**: Download sequences, logos, and analysis results
- **Real-time Updates**: Interactive parameter adjustment

## Project Structure

```
viz/
├── index.html              # Main visualizer application
├── bundle.js               # Core D3 visualization code
├── jaspar_meme.html        # JASPAR motif browser
├── memesuite-ui.html       # MEME Suite interface
├── vep.html                # Variant effect prediction viewer
├── logojs.html             # Standalone logo viewer
└── viz.md                  # Data format documentation
```

## Installation & Usage

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd viz
```

2. Open the main visualizer:
```bash
open index.html
# or
python -m http.server 8000  # then visit http://localhost:8000
```

### Loading Data

The visualizer supports HDF5 (.h5, .hdf5) and JSON data formats. See [viz.md](viz.md) for detailed data format specifications.

**Sampling Mode:**
- Load visualization data from unconditional/conditional generation
- Analyze the diffusion process without ground truth comparisons

**Evaluation Mode:**
- Load data with oracle predictions and ground truth labels
- Compare generated sequences to original test samples
- Track prediction quality across diffusion steps

### Data Format

Visualization data must include:
- **Metadata**: Dataset info, sequence length, total steps
- **Per-step data**: Sequences, score matrices, probability matrices
- **Optional**: Attribution scores, oracle predictions, ground truth data

See [viz.md](viz.md) for complete data structure documentation.

## Dependencies

### Core Libraries
- **D3.js** (v7.8.5): Data visualization and DOM manipulation
- **Chart.js** (v3.9.1): Interactive charts for metrics
- **h5wasm** (v0.4.9): HDF5 file reading in browser

### Integrated Packages
- **LogoJS**: SVG sequence logo generation (embedded)
- **memesuite-lite-js**: JavaScript port of FIMO and TOMTOM algorithms (see [github.com/54yyyu/memesuite-lite-js](https://github.com/54yyyu/memesuite-lite-js))
- **tfmodisco-lite-js**: TF-MoDISco motif discovery from importance scores (embedded)

## Example Workflows

### 1. Exploring Diffusion Sampling

```
1. Load visualization data (HDF5 or JSON)
2. Use timeline slider to navigate timesteps
3. Observe sequence evolution in the viewer
4. Analyze score/probability matrices
5. Export interesting sequences for further analysis
```

### 2. Motif Discovery Pipeline

```
1. Load evaluation data with attribution scores
2. Select sequences with high attribution
3. Run TF-MoDISco to discover patterns
4. Generate sequence logos for discovered motifs
5. Compare against JASPAR database using TOMTOM
6. Export results for downstream analysis
```

### 3. Variant Effect Analysis

```
1. Load VEP data with variant sequences
2. Compare wild-type vs variant predictions
3. Analyze attribution score differences
4. Identify functional variants
5. Export variant analysis results
```

## Data Generation

To generate sequences and visualization data, see the D3-DNA model repository: [github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion](https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion)

Example commands for generating visualization data from D3-DNA models:

**Sampling Mode:**
```bash
python model_zoo/deepstarr/sample.py \
    --checkpoint path/to/checkpoint.ckpt \
    --architecture transformer \
    --config model_zoo/deepstarr/configs/transformer.yaml \
    --num_samples 100 \
    --save_viz_data \
    --viz_format hdf5 \
    --viz_output my_sampling_viz.h5
```

**Evaluation Mode:**
```bash
python model_zoo/deepstarr/evaluate.py \
    --checkpoint path/to/checkpoint.ckpt \
    --oracle_checkpoint path/to/oracle.ckpt \
    --data_path model_zoo/deepstarr/DeepSTARR_data.h5 \
    --save_viz_data \
    --viz_output my_evaluation_viz.h5 \
    --max_samples 500 \
    --specific_indices "11,12,40"  # Optional: specific samples
```

See [viz.md](viz.md) for complete data generation instructions.

## Browser Compatibility

Tested and supported on:
- Chrome/Chromium (v90+)
- Firefox (v88+)
- Safari (v14+)
- Edge (v90+)

**Note:** HDF5 file loading requires WebAssembly support.

## Performance Considerations

- **Large datasets**: Files >500MB may take time to load
- **Memory usage**: Browser needs ~2-3x file size in RAM
- **Timesteps**: High step counts (>500) may impact slider performance
- **Recommendation**: Use JSON format for smaller datasets (<100MB)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization modes
- Performance optimizations for large datasets
- New motif analysis algorithms
- Export functionality enhancements

## License

This project includes multiple open-source components:
- LogoJS: MIT License
- memesuite-lite-js: MIT License
- tfmodisco-lite-js: MIT License

## Citations

If you use this visualizer in your research, please cite:

**D3-DNA:**
```
[D3-DNA paper citation - to be added]
```

**TF-MoDISco:**
```bibtex
@article{shrikumar2018technical,
  title={Technical note on transcription factor motif discovery from importance scores (tf-modisco) version 0.5.6.5},
  author={Shrikumar, Avanti and Greenside, Peyton and Kundaje, Anshul},
  journal={arXiv preprint arXiv:1811.00416},
  year={2018}
}
```

**MEME Suite:**
```bibtex
@article{bailey2015meme,
  title={The MEME Suite},
  author={Bailey, Timothy L and Johnson, James and Grant, Charles E and Noble, William S},
  journal={Nucleic acids research},
  volume={43},
  number={W1},
  pages={W39--W49},
  year={2015}
}
```

## Links

- [D3-DNA Model Repository](https://github.com/anirbansarkar-cs/D3-DNA-Discrete-Diffusion) - Generate sequences and visualization data
- [MemeSuite Lite JS](https://github.com/54yyyu/memesuite-lite-js) - FIMO and TOMTOM JavaScript implementation
- [LogoJS Website](http://logojs.wenglab.org/)
- [MEME Suite](https://meme-suite.org/)
- [JASPAR Database](http://jaspar.genereg.net/)
- [TF-MoDISco](https://github.com/kundajelab/tfmodisco)

Here’s a citation section tailored for your repository, following the format you provided:

---

### Cite

If you find **D3-Viz** helpful in your research, cite simply as:

```bibtex
@misc{d3viz,
  author = {Yiyang Yu},
  title = {D3-Viz: visualizer for DNA discrete diffusion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/54yyyu/d3-viz}
}
```

## Support

For issues and questions:
1. Check [viz.md](viz.md) for data format questions
2. For memesuite-lite-js issues, see [github.com/54yyyu/memesuite-lite-js](https://github.com/54yyyu/memesuite-lite-js)
3. Open an issue on the repository

---

**Note**: This visualizer is designed for research purposes and is under active development. Features and APIs may change.
