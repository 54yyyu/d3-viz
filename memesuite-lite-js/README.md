# memesuite-lite.js

[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)](https://www.javascript.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A JavaScript port of the Python [memesuite-lite](https://github.com/jmschrei/memesuite-lite) library, providing fast implementations of FIMO and TOMTOM algorithms for biological sequence analysis.

## Overview

This library implements core algorithms from the [MEME suite](https://meme-suite.org/meme/) in JavaScript:

- **FIMO** (Finding Individual Motif Instances) - Scan motifs against sequences
- **TOMTOM** - Compare motifs and calculate statistical significance
- **MEME file I/O** - Read and write MEME-formatted motif files
- **Utility functions** - One-hot encoding, sequence manipulation

## Installation

### Node.js
```bash
npm install memesuite-lite-js
```

### Browser
```html
<script src="memesuite-lite.js"></script>
```

## Quick Start

### Basic Usage

```javascript
// Import the library (Node.js)
const MemeSuiteLite = require('memesuite-lite-js');

// Or use in browser
// const MemeSuiteLite = window.MemeSuiteLite;

const msl = new MemeSuiteLite();

// One-hot encode a DNA sequence
const sequence = "ACGTACGT";
const oneHot = msl.oneHotEncode(sequence);
console.log(oneHot);
// [[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1]]

// Convert back to sequence
const backToSeq = msl.characters(oneHot);
console.log(backToSeq); // "ACGTACGT"
```

### Working with MEME Files

```javascript
const memeContent = `MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF TF1
letter-probability matrix: alength= 4 w= 4 nsites= 1 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.05 0.1 0.8 0.05
0.1 0.1 0.1 0.7
URL BLANK
`;

// Parse MEME file
const motifs = msl.readMeme(memeContent);
console.log(Object.keys(motifs)); // ["TF1"]
```

### FIMO - Finding Motif Instances

```javascript
// Run FIMO to find motif instances in sequences
const sequences = [
    "ATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTA",
    "TTTTAAAACCCCGGGG"
];

const fimoResults = msl.runFimo(memeContent, sequences);

fimoResults.forEach(result => {
    console.log(`Motif: ${result.motif_name}`);
    result.hits.forEach(hit => {
        console.log(`  Position ${hit.start}-${hit.end}: score=${hit.score.toFixed(3)}, p-value=${hit.p_value.toExponential(2)}`);
    });
});
```

### TOMTOM - Motif Comparison

```javascript
// Compare two sets of motifs
const queryMeme = `...`; // MEME format content
const targetMeme = `...`; // MEME format content

const tomtomResults = msl.runTomtom(queryMeme, targetMeme);

console.log('P-values:', tomtomResults.pValues);
console.log('Scores:', tomtomResults.scores);
console.log('Offsets:', tomtomResults.offsets);
```

## API Reference

### Core Methods

#### `oneHotEncode(sequence, alphabet, ignore)`
Convert a DNA sequence string to one-hot encoding.

**Parameters:**
- `sequence` (string): DNA sequence
- `alphabet` (Array): Characters to encode (default: ['A','C','G','T'])
- `ignore` (Array): Characters to ignore (default: ['N'])

**Returns:** 2D array representing one-hot encoding

#### `characters(pwm, alphabet, force)`
Convert PWM back to consensus sequence.

**Parameters:**
- `pwm` (Array): Position weight matrix
- `alphabet` (Array): Alphabet characters
- `force` (boolean): Force conversion even with ties

**Returns:** Consensus sequence string

### File I/O

#### `readMeme(fileContent, maxMotifs)`
Parse MEME format file content.

**Parameters:**
- `fileContent` (string): MEME file content
- `maxMotifs` (number): Maximum motifs to read

**Returns:** Object mapping motif names to PWMs

#### `writeMeme(motifs)`
Generate MEME format content from motifs.

**Parameters:**
- `motifs` (Object): Motif name to PWM mapping

**Returns:** MEME format string

### Algorithms

#### `fimo(motifs, sequences, options)`
Find Individual Motif Instances algorithm.

**Parameters:**
- `motifs` (Object): Motifs to search with
- `sequences` (Array): DNA sequences to search
- `options` (Object): Algorithm options
  - `threshold` (number): P-value threshold (default: 0.0001)
  - `reverseComplement` (boolean): Scan both strands (default: true)
  - `binSize` (number): Score discretization (default: 0.1)

**Returns:** Array of motif results with hits

#### `tomtom(queries, targets, options)`
TOMTOM motif comparison algorithm.

**Parameters:**
- `queries` (Array): Query PWMs
- `targets` (Array): Target PWMs
- `options` (Object): Algorithm options
  - `nScoreBins` (number): Score bins (default: 100)
  - `reverseComplement` (boolean): Compare both orientations (default: true)

**Returns:** Object with pValues, scores, offsets, overlaps, strands

### Convenience Methods

#### `runFimo(memeContent, sequences, options)`
Run FIMO with MEME file content.

#### `runTomtom(queryMemeContent, targetMemeContent, options)`
Run TOMTOM with MEME file contents.

## Examples

### Example 1: Basic Sequence Analysis

```javascript
const msl = new MemeSuiteLite();

// Create a simple motif
const motif = [
    [0.9, 0.1, 0.0, 0.0], // Strong A preference
    [0.0, 0.8, 0.1, 0.1], // Strong C preference  
    [0.1, 0.1, 0.7, 0.1], // Strong G preference
    [0.2, 0.2, 0.2, 0.4]  // Weak T preference
];

// Test sequences
const sequences = [
    "ACGTACGTACGT",
    "AAACCCGGGTTT", 
    "ATCGATCGATCG"
];

// Run FIMO
const results = msl.fimo({testMotif: motif}, sequences);
console.log('Found', results[0].hits.length, 'hits');
```

### Example 2: Working with Real JASPAR Motifs

```javascript
// You can download JASPAR motifs in MEME format and use them
const jasparContent = `...`; // Load from JASPAR database

const sequences = [
    "ATCGATCGATCGATCG",
    // ... more sequences
];

const results = msl.runFimo(jasparContent, sequences, {
    threshold: 0.001,
    reverseComplement: true
});

// Process results
results.forEach(motifResult => {
    if (motifResult.hits.length > 0) {
        console.log(`${motifResult.motif_name}: ${motifResult.hits.length} hits`);
    }
});
```

## Performance Notes

This JavaScript implementation provides:

- **Fast algorithms**: Core algorithms implemented with efficient JavaScript
- **Memory efficient**: Streaming processing where possible
- **Browser compatible**: Works in both Node.js and browser environments
- **No dependencies**: Pure JavaScript implementation

For extremely large-scale analyses (millions of sequences), consider using the original Python implementation which includes numba acceleration.

## Comparison to Python Version

| Feature | JavaScript | Python |
|---------|------------|---------|
| FIMO Algorithm | ✅ | ✅ |
| TOMTOM Algorithm | ✅ | ✅ |
| MEME I/O | ✅ | ✅ |
| Performance | Good | Excellent (numba) |
| Browser Support | ✅ | ❌ |
| Dependencies | None | numpy, numba, etc. |

## Contributing

This is a port of the excellent [memesuite-lite](https://github.com/jmschrei/memesuite-lite) Python library. Please consider:

1. Contributing to the original Python project
2. Reporting JavaScript-specific issues here
3. Submitting pull requests for improvements

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library, please cite the original MEME suite papers and consider citing:

```
@article{tomtom_lite_2025,
    title={tomtom-lite: Fast motif comparison for the modern age},
    author={Jacob Schreiber},
    journal={bioRxiv},
    year={2025}
}
```

## Links

- [Original Python Library](https://github.com/jmschrei/memesuite-lite)
- [MEME Suite](https://meme-suite.org/)
- [JASPAR Database](http://jaspar.genereg.net/)