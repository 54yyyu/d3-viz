/**
 * tfmodisco-lite.js - JavaScript implementation of TF-MoDISco motif discovery algorithm
 * 
 * A JavaScript port of the Python tfmodisco-lite library providing motif discovery
 * from neural network attribution scores.
 * 
 * Original Python library: https://github.com/jmschrei/tfmodisco-lite
 * 
 * @author JavaScript port of tfmodisco-lite
 * @version 1.0.0
 */

class TFModiscoLite {
    constructor() {
        this.version = '1.0.0';
    }

    // =====================================================
    // CORE DATA STRUCTURES
    // =====================================================

    /**
     * Seqlet class - represents a sequence segment with attribution scores
     */
    static Seqlet = class {
        constructor(exampleIdx, start, end, isRevcomp = false) {
            this.exampleIdx = exampleIdx;
            this.start = start;
            this.end = end;
            this.isRevcomp = isRevcomp;
            
            // These will be filled in by TrackSet.createSeqlets
            this.sequence = null;
            this.contribScores = null;
            this.hypotheticalContribs = null;
        }

        get length() {
            return this.end - this.start;
        }

        get string() {
            return `${this.exampleIdx}_${this.start}_${this.end}`;
        }

        /**
         * Create reverse complement of this seqlet
         * @returns {Seqlet} - New seqlet with reverse complement
         */
        revcomp() {
            const newSeqlet = new TFModiscoLite.Seqlet(
                this.exampleIdx, this.start, this.end, !this.isRevcomp
            );

            if (this.sequence) {
                newSeqlet.sequence = this._reverseComplementMatrix(this.sequence);
            }
            if (this.contribScores) {
                newSeqlet.contribScores = this._reverseComplementMatrix(this.contribScores);
            }
            if (this.hypotheticalContribs) {
                newSeqlet.hypotheticalContribs = this._reverseComplementMatrix(this.hypotheticalContribs);
            }

            return newSeqlet;
        }

        /**
         * Shift seqlet coordinates by offset
         * @param {number} shiftAmount - Amount to shift
         * @returns {Seqlet} - New shifted seqlet
         */
        shift(shiftAmount) {
            return new TFModiscoLite.Seqlet(
                this.exampleIdx,
                this.start + shiftAmount,
                this.end + shiftAmount,
                this.isRevcomp
            );
        }

        /**
         * Trim seqlet to new coordinates
         * @param {number} startIdx - New start index (relative)
         * @param {number} endIdx - New end index (relative)
         * @returns {Seqlet} - Trimmed seqlet
         */
        trim(startIdx, endIdx) {
            const newStart = this.isRevcomp ? this.end - endIdx : this.start + startIdx;
            const newEnd = this.isRevcomp ? this.end - startIdx : this.start + endIdx;

            const newSeqlet = new TFModiscoLite.Seqlet(
                this.exampleIdx, newStart, newEnd, this.isRevcomp
            );

            if (this.sequence) {
                newSeqlet.sequence = this.sequence.slice(startIdx, endIdx);
            }
            if (this.contribScores) {
                newSeqlet.contribScores = this.contribScores.slice(startIdx, endIdx);
            }
            if (this.hypotheticalContribs) {
                newSeqlet.hypotheticalContribs = this.hypotheticalContribs.slice(startIdx, endIdx);
            }

            return newSeqlet;
        }

        /**
         * Reverse complement a matrix [positions x bases]
         * @private
         */
        _reverseComplementMatrix(matrix) {
            const reversed = matrix.slice().reverse();
            return reversed.map(row => [row[3], row[2], row[1], row[0]]); // T,G,C,A
        }
    }

    /**
     * SeqletSet class - collection of seqlets forming a motif pattern
     */
    static SeqletSet = class {
        constructor(seqlets) {
            this.seqlets = seqlets || [];
            this.uniqueSeqlets = new Map();
            this.length = this.seqlets.length > 0 ? 
                Math.max(...this.seqlets.map(s => s.end - s.start)) : 0;
            
            // Pattern representations
            this.sequence = null; // Position frequency matrix
            this.contribScores = null; // Contribution weight matrix
            this.hypotheticalContribs = null; // Hypothetical contribution matrix
            this.perPositionCounts = null;
            
            // Subpattern information
            this.subclusters = null;
            this.subclusterToSubpattern = null;

            // Only initialize if seqlets have sequence data
            if (seqlets && seqlets.length > 0 && seqlets[0].sequence) {
                this._initializeMatrices();
                for (const seqlet of seqlets) {
                    if (!this.uniqueSeqlets.has(seqlet.string)) {
                        this._addSeqlet(seqlet);
                    }
                }
            }
        }

        /**
         * Initialize matrices for pattern representation
         * @private
         */
        _initializeMatrices() {
            this.sequence = Array(this.length).fill().map(() => [0, 0, 0, 0]);
            this.contribScores = Array(this.length).fill().map(() => [0, 0, 0, 0]);
            this.hypotheticalContribs = Array(this.length).fill().map(() => [0, 0, 0, 0]);
            this.perPositionCounts = Array(this.length).fill(0);
            
            this._sequenceSum = Array(this.length).fill().map(() => [0, 0, 0, 0]);
            this._contribSum = Array(this.length).fill().map(() => [0, 0, 0, 0]);
            this._hypotheticalSum = Array(this.length).fill().map(() => [0, 0, 0, 0]);
        }

        /**
         * Add a seqlet to the pattern
         * @private
         */
        _addSeqlet(seqlet) {
            const n = seqlet.length;
            
            this.seqlets.push(seqlet);
            this.uniqueSeqlets.set(seqlet.string, seqlet);
            
            // Update counts
            for (let i = 0; i < n; i++) {
                this.perPositionCounts[i] += 1.0;
            }

            // Add to running sums
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < 4; j++) {
                    this._sequenceSum[i][j] += seqlet.sequence[i][j];
                    this._contribSum[i][j] += seqlet.contribScores[i][j];
                    this._hypotheticalSum[i][j] += seqlet.hypotheticalContribs[i][j];
                }
            }

            // Update averages
            this._updateAverages();
        }

        /**
         * Update average matrices
         * @private
         */
        _updateAverages() {
            for (let i = 0; i < this.length; i++) {
                const count = Math.max(this.perPositionCounts[i], 1e-7);
                for (let j = 0; j < 4; j++) {
                    this.sequence[i][j] = this._sequenceSum[i][j] / count;
                    this.contribScores[i][j] = this._contribSum[i][j] / count;
                    this.hypotheticalContribs[i][j] = this._hypotheticalSum[i][j] / count;
                }
            }
        }

        /**
         * Copy this seqlet set
         * @returns {SeqletSet} - Copy of this set
         */
        copy() {
            return new TFModiscoLite.SeqletSet([...this.seqlets]);
        }

        /**
         * Trim pattern to positions with sufficient support
         * @param {number} minFrac - Minimum fraction of max support
         * @param {number} minNum - Minimum absolute number
         * @returns {SeqletSet} - Trimmed pattern
         */
        trimToSupport(minFrac, minNum) {
            const maxSupport = Math.max(...this.perPositionCounts);
            const threshold = Math.min(minNum, maxSupport * minFrac);
            
            let leftIdx = 0;
            while (leftIdx < this.length && this.perPositionCounts[leftIdx] < threshold) {
                leftIdx++;
            }

            let rightIdx = this.length;
            while (rightIdx > 0 && this.perPositionCounts[rightIdx - 1] < threshold) {
                rightIdx--;
            }
            
            return this.trimToIdx(leftIdx, rightIdx);
        }

        /**
         * Trim pattern to specific indices
         * @param {number} startIdx - Start index
         * @param {number} endIdx - End index
         * @returns {SeqletSet} - Trimmed pattern
         */
        trimToIdx(startIdx, endIdx) {
            const newSeqlets = this.seqlets.map(seqlet => 
                seqlet.trim(startIdx, endIdx)
            );
            return new TFModiscoLite.SeqletSet(newSeqlets);
        }
    }

    /**
     * TrackSet class - holds sequences and attribution data
     */
    static TrackSet = class {
        constructor(oneHot, contribScores, hypotheticalContribs) {
            this.oneHot = oneHot;              // [samples x positions x 4]
            this.contribScores = contribScores; // [samples x positions x 4]
            this.hypotheticalContribs = hypotheticalContribs; // [samples x positions x 4]
            this.length = oneHot[0].length;
        }

        /**
         * Create seqlets from coordinate specifications
         * @param {Array<Seqlet>} seqlets - Array of seqlet objects
         * @returns {Array<Seqlet>} - Seqlets with sequence data filled in
         */
        createSeqlets(seqlets) {
            for (const seqlet of seqlets) {
                const idx = seqlet.exampleIdx;
                const { start, end } = seqlet;

                if (seqlet.isRevcomp) {
                    // Reverse complement
                    seqlet.sequence = this._reverseComplementSlice(
                        this.oneHot[idx].slice(start, end)
                    );
                    seqlet.contribScores = this._reverseComplementSlice(
                        this.contribScores[idx].slice(start, end)
                    );
                    seqlet.hypotheticalContribs = this._reverseComplementSlice(
                        this.hypotheticalContribs[idx].slice(start, end)
                    );
                } else {
                    // Forward
                    seqlet.sequence = this.oneHot[idx].slice(start, end);
                    seqlet.contribScores = this.contribScores[idx].slice(start, end);
                    seqlet.hypotheticalContribs = this.hypotheticalContribs[idx].slice(start, end);
                }
            }

            return seqlets;
        }

        /**
         * Reverse complement a sequence slice
         * @private
         */
        _reverseComplementSlice(slice) {
            return slice.slice().reverse().map(row => [row[3], row[2], row[1], row[0]]);
        }
    }

    // =====================================================
    // UTILITY FUNCTIONS
    // =====================================================

    /**
     * Convert DNA sequence to one-hot encoding
     * @param {string} sequence - DNA sequence string
     * @param {Array<string>} alphabet - Base alphabet
     * @returns {Array<Array<number>>} - One-hot matrix [positions x bases]
     */
    oneHotEncode(sequence, alphabet = ['A', 'C', 'G', 'T']) {
        const n = sequence.length;
        
        // Return 4x0 matrix for empty sequence
        if (n === 0) {
            return [[], [], [], []];
        }
        
        const oneHot = Array(n).fill().map(() => Array(4).fill(0));
        
        const baseMap = new Map([['A', 0], ['C', 1], ['G', 2], ['T', 3]]);
        
        for (let i = 0; i < n; i++) {
            const base = sequence[i].toUpperCase();
            const idx = baseMap.get(base);
            if (idx !== undefined) {
                oneHot[i][idx] = 1;
            }
        }

        // Transpose to get [4 x n] format
        const result = [[], [], [], []];
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < 4; j++) {
                result[j].push(oneHot[i][j]);
            }
        }

        return result;
    }

    /**
     * Compute per-position information content
     * @param {Array<Array<number>>} ppm - Position probability matrix
     * @param {Array<number>} background - Background frequencies
     * @param {number} pseudocount - Pseudocount for smoothing
     * @returns {Array<number>} - Information content per position
     */
    computePerPositionIC(ppm, background = [0.25, 0.25, 0.25, 0.25], pseudocount = 0.001) {
        const alphabetLen = background.length;
        const ic = [];
        const seqLen = ppm[0] ? ppm[0].length : 0;

        for (let pos = 0; pos < seqLen; pos++) {
            let posIC = 0;
            let colSum = 0;
            
            // Calculate column sum for normalization
            for (let base = 0; base < alphabetLen; base++) {
                colSum += ppm[base][pos];
            }
            
            for (let base = 0; base < alphabetLen; base++) {
                const prob = (ppm[base][pos] + pseudocount) / (colSum + pseudocount * alphabetLen);
                if (prob > 0) {
                    posIC += prob * (Math.log2(prob) - Math.log2(background[base]));
                }
            }
            
            ic.push(posIC);
        }

        return ic;
    }

    /**
     * Convenience method for computePerPositionIC
     */
    computeIC(ppm, background = [0.25, 0.25, 0.25, 0.25], pseudocount = 0.001) {
        return this.computePerPositionIC(ppm, background, pseudocount);
    }

    /**
     * Generate IC-weighted consensus sequence
     */
    icWeightedConsensus(pwm) {
        const ic = this.computeIC(pwm);
        const alphabet = ['A', 'C', 'G', 'T'];
        let consensus = '';
        
        for (let pos = 0; pos < pwm[0].length; pos++) {
            let maxProb = 0;
            let bestBase = 0;
            
            for (let base = 0; base < 4; base++) {
                if (pwm[base][pos] > maxProb) {
                    maxProb = pwm[base][pos];
                    bestBase = base;
                }
            }
            
            // Use N for low information content positions
            if (ic[pos] < 0.5) {
                consensus += 'N';
            } else {
                consensus += alphabet[bestBase];
            }
        }
        
        return consensus;
    }

    /**
     * Simple matrix dot product
     */
    dotProduct(matA, matB) {
        const result = [];
        for (let i = 0; i < matA.length; i++) {
            const row = [];
            for (let j = 0; j < matB[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < matA[0].length; k++) {
                    sum += matA[i][k] * matB[k][j];
                }
                row.push(sum);
            }
            result.push(row);
        }
        return result;
    }

    /**
     * Convenience method for gapped k-mer representation
     */
    gappedKmerRepr(sequences, kmerLen, gapLen) {
        return this.cosineSimilarityFromSeqlets(
            sequences.map((seq, i) => {
                const seqlet = new TFModiscoLite.Seqlet(i, 0, seq.length);
                seqlet.sequence = seq;
                return seqlet;
            }), 
            100, 
            1, 
            { kmerLen, gapLen }
        );
    }

    /**
     * Direct cosine similarity between two seqlets
     */
    cosineSimilarity(seqlet1, seqlet2) {
        const seq1 = seqlet1.sequence.flat();
        const seq2 = seqlet2.sequence.flat();
        
        if (seq1.length !== seq2.length) return 0;
        
        const dotProd = seq1.reduce((sum, val, i) => sum + val * seq2[i], 0);
        const norm1 = Math.sqrt(seq1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(seq2.reduce((sum, val) => sum + val * val, 0));
        
        return norm1 && norm2 ? dotProd / (norm1 * norm2) : 0;
    }

    /**
     * Basic isotonic regression
     */
    isotonicRegression(values) {
        const result = values.slice();
        let changed = true;
        
        while (changed) {
            changed = false;
            for (let i = 0; i < result.length - 1; i++) {
                if (result[i] > result[i + 1]) {
                    const avg = (result[i] + result[i + 1]) / 2;
                    result[i] = avg;
                    result[i + 1] = avg;
                    changed = true;
                }
            }
        }
        
        return result;
    }

    /**
     * Sliding window sum operation
     * @param {Array<number>} arr - Input array
     * @param {number} windowSize - Window size
     * @returns {Array<number>} - Windowed sums
     */
    slidingWindowSum(arr, windowSize) {
        const result = [];
        
        for (let i = 0; i <= arr.length - windowSize; i++) {
            let sum = 0;
            for (let j = 0; j < windowSize; j++) {
                sum += arr[i + j];
            }
            result.push(sum);
        }

        return result;
    }

    /**
     * Binary search for perplexity in t-SNE-like algorithms
     * @param {number} desiredPerplexity - Target perplexity
     * @param {Array<number>} distances - Distance array
     * @returns {number} - Beta parameter
     */
    binarySearchPerplexity(desiredPerplexity, distances) {
        const EPSILON_DBL = 1e-8;
        const PERPLEXITY_TOLERANCE = 1e-5;
        const nSteps = 100;
        
        const desiredEntropy = Math.log(desiredPerplexity);
        
        let betaMin = -Infinity;
        let betaMax = Infinity;
        let beta = 1.0;
        
        for (let l = 0; l < nSteps; l++) {
            const ps = distances.map(d => Math.exp(-d * beta));
            const sumPs = ps.reduce((sum, p) => sum + p, 0) + 1;
            const normalizedPs = ps.map(p => p / Math.max(sumPs, EPSILON_DBL));
            const sumDistiPi = distances.reduce((sum, d, i) => sum + d * normalizedPs[i], 0);
            const entropy = Math.log(sumPs) + beta * sumDistiPi;
            
            const entropyDiff = entropy - desiredEntropy;
            if (Math.abs(entropyDiff) <= PERPLEXITY_TOLERANCE) {
                break;
            }
            
            if (entropyDiff > 0.0) {
                betaMin = beta;
                if (betaMax === Infinity) {
                    beta *= 2.0;
                } else {
                    beta = (beta + betaMax) / 2.0;
                }
            } else {
                betaMax = beta;
                if (betaMin === -Infinity) {
                    beta /= 2.0;
                } else {
                    beta = (beta + betaMin) / 2.0;
                }
            }
        }
        
        return beta;
    }

    /**
     * Create sparse matrix representation
     * @param {Array<number>} data - Non-zero values
     * @param {Array<number>} rows - Row indices
     * @param {Array<number>} cols - Column indices
     * @param {number} nRows - Number of rows
     * @param {number} nCols - Number of columns
     * @returns {Object} - Sparse matrix object
     */
    createSparseMatrix(data, rows, cols, nRows, nCols) {
        return {
            data: data.slice(),
            rows: rows.slice(),
            cols: cols.slice(),
            shape: [nRows, nCols],
            nnz: data.length
        };
    }

    /**
     * Multiply sparse matrix by vector
     * @param {Object} matrix - Sparse matrix
     * @param {Array<number>} vector - Dense vector
     * @returns {Array<number>} - Result vector
     */
    sparseMatrixVectorMultiply(matrix, vector) {
        const result = Array(matrix.shape[0]).fill(0);
        
        for (let i = 0; i < matrix.nnz; i++) {
            result[matrix.rows[i]] += matrix.data[i] * vector[matrix.cols[i]];
        }
        
        return result;
    }

    /**
     * Compute rolling window with given window size
     * @param {Array<number>} array - Input array
     * @param {number} window - Window size
     * @returns {Array<Array<number>>} - Rolling windows
     */
    rollingWindow(array, window) {
        const result = [];
        for (let i = 0; i <= array.length - window; i++) {
            result.push(array.slice(i, i + window));
        }
        return result;
    }

    /**
     * Compute cumulative sum of array
     * @param {Array<number>} array - Input array
     * @returns {Array<number>} - Cumulative sum
     */
    cumulativeSum(array) {
        const result = [];
        let sum = 0;
        for (const val of array) {
            sum += val;
            result.push(sum);
        }
        return result;
    }

    /**
     * Get 2D data from patterns for similarity calculations
     * @param {Array<Object>} patterns - Array of patterns or seqlets
     * @param {string} transformer - Transformation to apply ('l1' or 'magnitude')
     * @param {boolean} includeHypothetical - Whether to include hypothetical contributions
     * @returns {Object} - {fwdData, revData}
     */
    get2dDataFromPatterns(patterns, transformer = 'l1', includeHypothetical = true) {
        const func = transformer === 'l1' ? this._l1Normalize : this._magnitudeNormalize;
        const tracks = includeHypothetical ? 
            ['hypotheticalContribs', 'contribScores'] : ['contribScores'];

        const allFwdData = [];
        const allRevData = [];

        for (const pattern of patterns) {
            const snippets = tracks.map(track => pattern[track]);
            
            const fwdData = this._concatenateAndNormalize(snippets, func);
            const revData = this._concatenateAndNormalize(
                snippets.map(snippet => this._reverseComplementMatrix(snippet)), func
            );

            allFwdData.push(fwdData);
            allRevData.push(revData);
        }

        return { fwdData: allFwdData, revData: allRevData };
    }

    /**
     * L1 normalize matrix
     * @private
     */
    _l1Normalize(matrix) {
        const flattened = matrix.flat();
        const absSum = flattened.reduce((sum, val) => sum + Math.abs(val), 0);
        if (absSum === 0) return matrix;
        
        return matrix.map(row => row.map(val => val / absSum));
    }

    /**
     * Magnitude normalize matrix
     * @private
     */
    _magnitudeNormalize(matrix) {
        const flattened = matrix.flat();
        const mean = flattened.reduce((sum, val) => sum + val, 0) / flattened.length;
        const centered = flattened.map(val => val - mean);
        const norm = Math.sqrt(centered.reduce((sum, val) => sum + val * val, 0)) + 1e-7;
        
        return matrix.map(row => row.map(val => (val - mean) / norm));
    }

    /**
     * Concatenate and normalize multiple matrices
     * @private
     */
    _concatenateAndNormalize(matrices, normalizeFunc) {
        const concatenated = matrices.reduce((acc, matrix) => {
            return acc.map((row, i) => row.concat(matrix[i] || []));
        }, matrices[0].map(row => [...row]));
        
        return normalizeFunc.call(this, concatenated);
    }

    /**
     * Reverse complement a matrix
     * @private
     */
    _reverseComplementMatrix(matrix) {
        return matrix.slice().reverse().map(row => [row[3], row[2], row[1], row[0]]);
    }

    /**
     * Compute Pearson correlation between two arrays
     * @param {Array<number>} x - First array
     * @param {Array<number>} y - Second array
     * @returns {number} - Correlation coefficient
     */
    pearsonCorrelation(x, y) {
        if (x.length !== y.length) {
            throw new Error('Arrays must have same length');
        }

        const n = x.length;
        const meanX = x.reduce((sum, val) => sum + val, 0) / n;
        const meanY = y.reduce((sum, val) => sum + val, 0) / n;

        let numerator = 0;
        let sumXSquared = 0;
        let sumYSquared = 0;

        for (let i = 0; i < n; i++) {
            const xDiff = x[i] - meanX;
            const yDiff = y[i] - meanY;
            numerator += xDiff * yDiff;
            sumXSquared += xDiff * xDiff;
            sumYSquared += yDiff * yDiff;
        }

        const denominator = Math.sqrt(sumXSquared * sumYSquared);
        return denominator === 0 ? 0 : numerator / denominator;
    }

    /**
     * Find mode of array using binning
     * @param {Array<number>} values - Input values
     * @param {number} bins - Number of bins
     * @returns {Object} - {leftEdge, rightEdge, modeValues}
     */
    binMode(values, bins = 1000) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binSize = (max - min) / bins;
        
        const counts = Array(bins).fill(0);
        for (const val of values) {
            const binIdx = Math.min(Math.floor((val - min) / binSize), bins - 1);
            counts[binIdx]++;
        }

        const peakBin = counts.indexOf(Math.max(...counts));
        const leftEdge = min + peakBin * binSize;
        const rightEdge = min + (peakBin + 1) * binSize;
        const modeValues = values.filter(val => val >= leftEdge && val < rightEdge);

        return { leftEdge, rightEdge, modeValues };
    }

    /**
     * Generate Laplacian null distribution
     * @param {Array<Array<number>>} track - Attribution track
     * @param {number} windowSize - Window size
     * @param {number} numToSample - Number of samples to generate
     * @param {number} randomSeed - Random seed
     * @returns {Object} - {posValues, negValues}
     */
    laplacianNull(track, windowSize, numToSample, randomSeed = 1234) {
        const percentilesToUse = Array.from({length: 19}, (_, i) => 5 * (i + 1));
        
        // Flatten all values
        const values = track.flat();
        
        // Estimate mu using two-level histogram
        const firstMode = this.binMode(values);
        const secondMode = this.binMode(firstMode.modeValues);
        const mu = (secondMode.leftEdge + secondMode.rightEdge) / 2;

        const posValues = values.filter(v => v >= mu);
        const negValues = values.filter(v => v <= mu);

        // Calculate lambda parameters
        let posLaplaceLambda = 0;
        for (const p of percentilesToUse) {
            const percentile = this._percentile(posValues, p);
            posLaplaceLambda = Math.max(posLaplaceLambda, 
                -Math.log(1 - p/100) / (percentile - mu));
        }

        let negLaplaceLambda = 0;
        for (const p of percentilesToUse) {
            const percentile = this._percentile(negValues, 100 - p);
            negLaplaceLambda = Math.max(negLaplaceLambda,
                -Math.log(1 - p/100) / Math.abs(percentile - mu));
        }

        // Generate samples
        const rng = this._seedRandom(randomSeed);
        const probPos = posValues.length / (posValues.length + negValues.length);
        const sampledVals = [];

        for (let i = 0; i < numToSample; i++) {
            const sign = rng() < probPos ? 1 : -1;
            const sampledCdf = rng();
            
            let val;
            if (sign === 1) {
                val = -Math.log(1 - sampledCdf) / posLaplaceLambda + mu;
            } else {
                val = mu + Math.log(1 - sampledCdf) / negLaplaceLambda;
            }
            sampledVals.push(val);
        }

        return {
            posValues: sampledVals.filter(v => v >= 0),
            negValues: sampledVals.filter(v => v < 0)
        };
    }

    /**
     * Calculate percentile of array
     * @private
     */
    _percentile(arr, p) {
        const sorted = arr.slice().sort((a, b) => a - b);
        const index = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        
        if (lower === upper) {
            return sorted[lower];
        }
        
        const weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    /**
     * Simple seeded random number generator
     * @private
     */
    _seedRandom(seed) {
        let state = seed;
        return function() {
            state = (state * 1664525 + 1013904223) % Math.pow(2, 32);
            return state / Math.pow(2, 32);
        };
    }

    /**
     * Isotonic regression for threshold calculation
     * @param {Array<number>} values - Values to fit
     * @param {Array<number>} nullValues - Null distribution values
     * @param {boolean} increasing - Whether to fit increasing function
     * @param {number} targetFDR - Target false discovery rate
     * @param {number} minFracNeg - Minimum fraction negative
     * @returns {number} - Threshold value
     */
    isotonicThreshold(values, nullValues, increasing, targetFDR, minFracNeg = 0.95) {
        const n1 = values.length;
        const n2 = nullValues.length;
        
        // Combine values and create labels
        const X = [...values, ...nullValues];
        const y = [...Array(n1).fill(1), ...Array(n2).fill(0)];
        
        // Sample weights
        const w = n1 / n2;
        const sampleWeight = [...Array(n1).fill(1), ...Array(n2).fill(w)];
        
        // Simple isotonic regression implementation
        const model = this._simpleIsotonicRegression(X, y, sampleWeight, increasing);
        
        // Calculate threshold
        const minPrecX = increasing ? Math.min(...X) : Math.max(...X);
        const minPrecision = this._isotonicTransform(model, minPrecX);
        
        let impliedFracNeg = -1 / (1 - (1 / Math.max(minPrecision, 1e-7)));
        if (impliedFracNeg > 1.0 || impliedFracNeg < minFracNeg) {
            impliedFracNeg = Math.max(Math.min(1.0, impliedFracNeg), minFracNeg);
        }
        
        const precisions = values.map(val => {
            const transformed = this._isotonicTransform(model, val);
            return Math.min(Math.max(1 + impliedFracNeg * (1 - (1 / Math.max(transformed, 1e-7))), 0.0), 1.0);
        });
        
        // Find threshold where precision >= (1 - targetFDR)
        const threshold = 1 - targetFDR;
        for (let i = 0; i < values.length; i++) {
            if (precisions[i] >= threshold) {
                return values[i];
            }
        }
        
        return values[values.length - 1];
    }

    /**
     * Simple isotonic regression implementation
     * @private
     */
    _simpleIsotonicRegression(X, y, weights, increasing) {
        // Simplified implementation - in practice would use PAVA algorithm
        const points = X.map((x, i) => ({ x, y: y[i], weight: weights[i] }));
        points.sort((a, b) => increasing ? a.x - b.x : b.x - a.x);
        
        return { points, increasing };
    }

    /**
     * Transform value using isotonic model
     * @private
     */
    _isotonicTransform(model, value) {
        // Simple linear interpolation
        const { points } = model;
        
        if (value <= points[0].x) return points[0].y;
        if (value >= points[points.length - 1].x) return points[points.length - 1].y;
        
        for (let i = 0; i < points.length - 1; i++) {
            if (value >= points[i].x && value <= points[i + 1].x) {
                const t = (value - points[i].x) / (points[i + 1].x - points[i].x);
                return points[i].y + t * (points[i + 1].y - points[i].y);
            }
        }
        
        return points[points.length - 1].y;
    }

    // =====================================================
    // GAPPED K-MER AND SIMILARITY CALCULATIONS
    // =====================================================

    /**
     * Extract gapped k-mers from seqlets
     * @param {Array<Object>} seqlets - Array of seqlets
     * @param {number} topN - Top N positions to consider
     * @param {number} minK - Minimum k-mer length
     * @param {number} maxK - Maximum k-mer length
     * @param {number} maxGap - Maximum gap size
     * @param {number} maxLen - Maximum total length
     * @param {number} maxEntries - Maximum number of k-mers per seqlet
     * @param {boolean} takeFwd - Whether to take forward direction
     * @param {number} sign - Sign for attribution scores (+1 or -1)
     * @returns {Object} - Sparse matrix representation
     */
    extractGappedKmers(seqlets, topN = 20, minK = 4, maxK = 6, maxGap = 15, maxLen = 15, 
                       maxEntries = 500, takeFwd = true, sign = 1) {
        
        const gkmerData = [];
        
        for (const seqlet of seqlets) {
            let oneHot = seqlet.sequence;
            let contribScores = seqlet.hypotheticalContribs;
            
            if (!takeFwd) {
                oneHot = this._reverseComplementMatrix(oneHot);
                contribScores = this._reverseComplementMatrix(contribScores);
            }
            
            // Apply sign and multiply with one-hot encoding
            const signedContribs = contribScores.map((row, i) => 
                row.map((val, j) => val * oneHot[i][j] * sign)
            );
            
            // Get per-position importance and base calls
            const perPosImp = signedContribs.map(row => row.reduce((sum, val) => sum + val, 0));
            const perPosBases = oneHot.map(row => row.indexOf(Math.max(...row)));
            
            // Get top N positions
            const positions = Array.from({length: perPosImp.length}, (_, i) => i);
            positions.sort((a, b) => perPosImp[b] - perPosImp[a]);
            const topPositions = positions.slice(0, topN);
            
            // Create position-base-importance triples
            const positionData = topPositions
                .map(pos => [pos, perPosBases[pos], perPosImp[pos]])
                .sort((a, b) => a[0] - b[0]);
            
            // Extract gapped k-mers
            const gkmers = this._extractGkmersFromPositions(
                positionData, minK, maxK, maxGap, maxLen, maxEntries
            );
            
            gkmerData.push(gkmers);
        }
        
        return this._createSparseMatrixFromGkmers(gkmerData, Math.pow(5, maxLen));
    }

    /**
     * Extract gapped k-mers from position data
     * @private
     */
    _extractGkmersFromPositions(positionData, minK, maxK, maxGap, maxLen, maxEntries) {
        const gkmerAttrs = new Map();
        const n = positionData.length;
        
        // Initialize with single positions
        let lastKGkmers = [];
        let lastKGkmersAttrs = [];
        let lastKGkmersHashes = [];
        
        for (let i = 0; i < n; i++) {
            const [pos, base, attr] = positionData[i];
            lastKGkmers.push([i]);
            lastKGkmersAttrs.push([attr]);
            lastKGkmersHashes.push([base + 1]); // +1 to avoid 0
        }
        
        // Build k-mers of increasing length
        for (let k = 2; k <= maxK; k++) {
            const newGkmers = [];
            const newGkmersAttrs = [];
            const newGkmersHashes = [];
            
            for (let j = 0; j < n; j++) {
                const startPos = positionData[j][0];
                const currentGkmers = [];
                const currentAttrs = [];
                const currentHashes = [];
                
                for (let i = j + 1; i < n; i++) {
                    const [pos, base, attr] = positionData[i];
                    
                    if (pos - startPos >= maxLen) break;
                    
                    for (let g = 0; g < lastKGkmers[j].length; g++) {
                        const gkmer = lastKGkmers[j][g];
                        const gkmerAttr = lastKGkmersAttrs[j][g];
                        const gkmerHash = lastKGkmersHashes[j][g];
                        
                        const lastPos = positionData[gkmer][0];
                        if (lastPos >= pos) break;
                        if (pos - lastPos > maxGap) continue;
                        
                        const length = pos - startPos;
                        const newGkmerHash = gkmerHash + (base + 1) * Math.pow(5, length);
                        const newGkmerAttr = gkmerAttr + attr;
                        
                        currentGkmers.push(i);
                        currentAttrs.push(newGkmerAttr);
                        currentHashes.push(newGkmerHash);
                        
                        if (k >= minK) {
                            const avgAttr = newGkmerAttr / k;
                            gkmerAttrs.set(newGkmerHash, 
                                (gkmerAttrs.get(newGkmerHash) || 0) + avgAttr);
                        }
                    }
                }
                
                newGkmers[j] = currentGkmers;
                newGkmersAttrs[j] = currentAttrs;
                newGkmersHashes[j] = currentHashes;
            }
            
            lastKGkmers = newGkmers;
            lastKGkmersAttrs = newGkmersAttrs;
            lastKGkmersHashes = newGkmersHashes;
        }
        
        // Convert to arrays and sort by absolute value
        const keys = Array.from(gkmerAttrs.keys());
        const scores = keys.map(key => gkmerAttrs.get(key));
        
        const sortedIndices = scores
            .map((score, i) => [Math.abs(score), i])
            .sort((a, b) => b[0] - a[0])
            .slice(0, maxEntries)
            .map(([_, i]) => i);
        
        return {
            keys: sortedIndices.map(i => keys[i]),
            scores: sortedIndices.map(i => scores[i])
        };
    }

    /**
     * Create sparse matrix from gapped k-mer data
     * @private
     */
    _createSparseMatrixFromGkmers(gkmerData, vocabSize) {
        const nRows = gkmerData.length;
        const rowIndices = [];
        const colIndices = [];
        const data = [];
        
        for (let i = 0; i < nRows; i++) {
            const { keys, scores } = gkmerData[i];
            for (let j = 0; j < keys.length; j++) {
                rowIndices.push(i);
                colIndices.push(keys[j]);
                data.push(scores[j]);
            }
        }
        
        return this.createSparseMatrix(data, rowIndices, colIndices, nRows, vocabSize);
    }

    /**
     * Calculate cosine similarity from seqlets using gapped k-mers
     * @param {Array<Object>} seqlets - Array of seqlets
     * @param {number} nNeighbors - Number of neighbors to find
     * @param {number} sign - Sign for attribution scores
     * @param {Object} gkmerParams - Gapped k-mer parameters
     * @returns {Object} - {similarities, neighbors}
     */
    cosineSimilarityFromSeqlets(seqlets, nNeighbors, sign, gkmerParams = {}) {
        const params = {
            topN: 20,
            minK: 4,
            maxK: 6,
            maxGap: 15,
            maxLen: 15,
            maxEntries: 500,
            ...gkmerParams
        };

        // Extract forward and reverse gapped k-mers
        const XFwd = this.extractGappedKmers(seqlets, params.topN, params.minK, 
            params.maxK, params.maxGap, params.maxLen, params.maxEntries, true, sign);
        
        const XBwd = this.extractGappedKmers(seqlets, params.topN, params.minK, 
            params.maxK, params.maxGap, params.maxLen, params.maxEntries, false, sign);

        // L2 normalize the sparse matrices
        const XFwdNorm = this._l2NormalizeSparseMatrix(XFwd);
        const YBwdNorm = this._l2NormalizeSparseMatrix(XBwd);
        
        const n = seqlets.length;
        const k = Math.min(nNeighbors + 1, n);
        
        return this._sparseMatrixDot(XFwdNorm, YBwdNorm, k);
    }

    /**
     * L2 normalize sparse matrix rows
     * @private
     */
    _l2NormalizeSparseMatrix(sparseMatrix) {
        const { data, rows, cols, shape } = sparseMatrix;
        const nRows = shape[0];
        
        // Calculate row norms
        const rowNorms = Array(nRows).fill(0);
        for (let i = 0; i < data.length; i++) {
            rowNorms[rows[i]] += data[i] * data[i];
        }
        
        for (let i = 0; i < nRows; i++) {
            rowNorms[i] = Math.sqrt(rowNorms[i]) || 1; // Avoid division by zero
        }
        
        // Normalize data
        const normalizedData = data.map((val, i) => val / rowNorms[rows[i]]);
        
        return this.createSparseMatrix(normalizedData, rows, cols, shape[0], shape[1]);
    }

    /**
     * Compute sparse matrix multiplication for similarity
     * @private
     */
    _sparseMatrixDot(X, Y, k) {
        const nRows = X.shape[0];
        const similarities = Array(nRows).fill().map(() => Array(k).fill(0));
        const neighbors = Array(nRows).fill().map(() => Array(k).fill(0));
        
        for (let i = 0; i < nRows; i++) {
            const dot = Array(nRows).fill(0);
            
            // Calculate dot products for row i
            for (let idx = 0; idx < X.data.length; idx++) {
                if (X.rows[idx] === i) {
                    const col = X.cols[idx];
                    const xVal = X.data[idx];
                    
                    // Find corresponding entries in Y
                    for (let jdx = 0; jdx < Y.data.length; jdx++) {
                        if (Y.cols[jdx] === col) {
                            const j = Y.rows[jdx];
                            dot[j] += xVal * Y.data[jdx];
                        }
                    }
                }
            }
            
            // Also compute X dot X for this row (forward-forward similarity)
            for (let idx1 = 0; idx1 < X.data.length; idx1++) {
                if (X.rows[idx1] === i) {
                    const col = X.cols[idx1];
                    const xVal1 = X.data[idx1];
                    
                    for (let idx2 = 0; idx2 < X.data.length; idx2++) {
                        if (X.cols[idx2] === col) {
                            const j = X.rows[idx2];
                            dot[j] = Math.max(dot[j], xVal1 * X.data[idx2]);
                        }
                    }
                }
            }
            
            // Sort and take top k
            const dotWithIndices = dot.map((val, idx) => [val, idx]);
            dotWithIndices.sort((a, b) => b[0] - a[0]);
            
            for (let j = 0; j < k; j++) {
                similarities[i][j] = dotWithIndices[j][0];
                neighbors[i][j] = dotWithIndices[j][1];
            }
        }
        
        return { similarities, neighbors };
    }

    // =====================================================
    // JACCARD SIMILARITY WITH ALIGNMENT
    // =====================================================

    /**
     * Calculate Jaccard similarity from seqlets with alignment
     * @param {Array<Object>} seqlets - Array of seqlets
     * @param {number} minOverlap - Minimum overlap fraction
     * @param {Array<Object>} filterSeqlets - Seqlets to filter against (optional)
     * @param {Array<Array<number>>} seqletNeighbors - Neighbor indices for each seqlet
     * @returns {Array<Array<number>>} - Affinity matrix
     */
    jaccardFromSeqlets(seqlets, minOverlap, filterSeqlets = null, seqletNeighbors = null) {
        const { fwdData, revData } = this.get2dDataFromPatterns(seqlets);
        let filtersFwdData, filtersRevData;

        if (filterSeqlets === null) {
            filterSeqlets = seqlets;
            filtersFwdData = fwdData;
            filtersRevData = revData;
        } else {
            const filterData = this.get2dDataFromPatterns(filterSeqlets);
            filtersFwdData = filterData.fwdData;
            filtersRevData = filterData.revData;
        }

        if (seqletNeighbors === null) {
            seqletNeighbors = seqlets.map(() => 
                Array.from({length: filterSeqlets.length}, (_, i) => i)
            );
        }

        // Apply cross metric for forward patterns
        const affmatFwd = this.jaccard(
            filtersFwdData, fwdData, minOverlap, seqletNeighbors, Math.ceil, true
        );

        // Apply cross metric for reverse patterns  
        const affmatRev = this.jaccard(
            filtersRevData, fwdData, minOverlap, seqletNeighbors, Math.ceil, true
        );

        // Take maximum of forward and reverse similarities
        const affmat = [];
        for (let i = 0; i < affmatFwd.length; i++) {
            affmat[i] = [];
            for (let j = 0; j < affmatFwd[i].length; j++) {
                affmat[i][j] = Math.max(affmatFwd[i][j], affmatRev[i][j]);
            }
        }

        return affmat;
    }

    /**
     * Calculate Jaccard similarity with alignment
     * @param {Array<Array<Array<number>>>} X - Query patterns [patterns x positions x features]
     * @param {Array<Array<Array<number>>>} Y - Target patterns [patterns x positions x features]
     * @param {number} minOverlap - Minimum overlap fraction (optional)
     * @param {Array<Array<number>>} seqletNeighbors - Neighbor indices for each pattern
     * @param {Function} func - Function to calculate padding (Math.ceil)
     * @param {boolean} returnSparse - Whether to return sparse format (max scores only)
     * @returns {Array<Array<number>>|Object} - Similarity results
     */
    jaccard(X, Y, minOverlap = null, seqletNeighbors = null, func = Math.ceil, returnSparse = false) {
        if (seqletNeighbors === null) {
            seqletNeighbors = Y.map(() => Array.from({length: X.length}, (_, i) => i));
        }

        let nPad = 0;
        let paddedY = Y;

        if (minOverlap !== null) {
            nPad = func(X[0].length * (1 - minOverlap));
            paddedY = Y.map(pattern => {
                const padding = Array(nPad).fill().map(() => Array(pattern[0].length).fill(0));
                return [...padding, ...pattern, ...padding];
            });
        }

        const lenOutput = 1 + paddedY[0].length - X[0].length;
        const scores = Array(paddedY.length).fill().map(() => 
            Array(seqletNeighbors[0].length).fill().map(() => 
                Array(lenOutput).fill(0)
            )
        );

        // Convert to Float32 for consistency with original
        const XFloat = X.map(pattern => 
            pattern.map(pos => pos.map(val => parseFloat(val)))
        );
        const YFloat = paddedY.map(pattern => 
            pattern.map(pos => pos.map(val => parseFloat(val)))
        );

        this._jaccardCore(XFloat, YFloat, seqletNeighbors, scores);

        if (returnSparse) {
            return scores.map(row => row.map(col => Math.max(...col)));
        }

        // Find best alignments
        const results = Array(paddedY.length).fill().map(() => 
            Array(seqletNeighbors[0].length).fill().map(() => [0, 0])
        );

        for (let i = 0; i < paddedY.length; i++) {
            for (let j = 0; j < seqletNeighbors[0].length; j++) {
                const maxIdx = scores[i][j].indexOf(Math.max(...scores[i][j]));
                results[i][j][0] = scores[i][j][maxIdx];
                results[i][j][1] = maxIdx - nPad;
            }
        }

        return results;
    }

    /**
     * Core Jaccard calculation with alignment scanning
     * @private
     */
    _jaccardCore(X, Y, neighbors, scores) {
        const nx = X.length;
        const d = X[0].length;
        const m = X[0][0].length;
        const ny = Y.length;
        const lenOutput = scores[0][0].length;

        for (let l = 0; l < ny; l++) {
            for (let idx = 0; idx < lenOutput; idx++) {
                for (let i = 0; i < neighbors[l].length; i++) {
                    let minSum = 0.0;
                    let maxSum = 0.0;
                    const neighborLi = neighbors[l][i];

                    for (let j = 0; j < d; j++) {
                        const jIdx = j;
                        const yIdx = idx + j;

                        if (yIdx >= Y[l].length) continue;

                        for (let k = 0; k < m; k++) {
                            const xVal = X[neighborLi][jIdx][k];
                            const yVal = Y[l][yIdx][k];
                            
                            const sign = Math.sign(xVal) * Math.sign(yVal);
                            const absX = Math.abs(xVal);
                            const absY = Math.abs(yVal);

                            if (absY > absX) {
                                minSum += absX * sign;
                                maxSum += absY;
                            } else {
                                minSum += absY * sign;
                                maxSum += absX;
                            }
                        }
                    }

                    scores[l][i][idx] = maxSum > 0 ? minSum / maxSum : 0;
                }
            }
        }
    }

    /**
     * Calculate pairwise Jaccard similarity efficiently
     * @param {Array<Array<number>>} X - Input matrix [samples x features]
     * @param {number} k - Number of top neighbors to keep
     * @returns {Object} - {jaccards, neighbors}
     */
    pairwiseJaccard(X, k) {
        const n = X.length;
        const m = X[0].length;

        const jaccards = Array(n).fill().map(() => Array(k).fill(0));
        const neighbors = Array(n).fill().map(() => Array(k).fill(0));

        for (let i = 0; i < n; i++) {
            const jaccardScores = Array(n).fill(0);

            for (let j = 0; j < n; j++) {
                let minSum = 0.0;
                let maxSum = 0.0;

                for (let l = 0; l < m; l++) {
                    const sign = Math.sign(X[i][l]) * Math.sign(X[j][l]);
                    const xi = Math.abs(X[i][l]);
                    const xj = Math.abs(X[j][l]);

                    if (xi > xj) {
                        minSum += xj * sign;
                        maxSum += xi;
                    } else {
                        minSum += xi * sign;
                        maxSum += xj;
                    }
                }

                jaccardScores[j] = maxSum > 0 ? minSum / maxSum : 0;
            }

            // Sort and take top k
            const indices = Array.from({length: n}, (_, idx) => idx);
            indices.sort((a, b) => jaccardScores[b] - jaccardScores[a]);

            for (let j = 0; j < k; j++) {
                jaccards[i][j] = jaccardScores[indices[j]];
                neighbors[i][j] = indices[j];
            }
        }

        return { jaccards, neighbors };
    }

    /**
     * Calculate Pearson correlation with alignment (for pattern comparison)
     * @param {Array<Array<Array<number>>>} X - Query patterns
     * @param {Array<Array<Array<number>>>} Y - Target patterns
     * @param {number} minOverlap - Minimum overlap fraction
     * @param {Function} func - Padding calculation function
     * @returns {Array<Array<number>>} - Correlation scores and offsets
     */
    pearsonCorrelationWithAlignment(X, Y, minOverlap = null, func = Math.ceil) {
        if (X.length === 0 || Y.length === 0) return [];

        // Ensure 3D input [patterns x positions x features]
        const X3d = X.length > 0 && !Array.isArray(X[0][0]) ? [X] : X;
        const Y3d = Y.length > 0 && !Array.isArray(Y[0][0]) ? [Y] : Y;

        let nPad = 0;
        let paddedY = Y3d;

        if (minOverlap !== null) {
            nPad = func(X3d[0].length * (1 - minOverlap));
            paddedY = Y3d.map(pattern => {
                const padding = Array(nPad).fill().map(() => Array(pattern[0].length).fill(0));
                return [...padding, ...pattern, ...padding];
            });
        }

        const d = X3d[0].length;
        const lenOutput = 1 + paddedY[0].length - d;
        const scores = Array(lenOutput).fill(0);

        for (let idx = 0; idx < lenOutput; idx++) {
            const YSlice = paddedY[0].slice(idx, idx + d);

            // Flatten for correlation calculation
            const xFlat = X3d[0].flat(2);
            const yFlat = YSlice.flat(2);

            // Calculate norms
            const xNorm = Math.sqrt(xFlat.reduce((sum, val) => sum + val * val, 0));
            const yNorm = Math.sqrt(yFlat.reduce((sum, val) => sum + val * val, 0));

            if (xNorm === 0 || yNorm === 0) {
                scores[idx] = 0;
            } else {
                const dot = xFlat.reduce((sum, val, i) => sum + val * yFlat[i], 0);
                scores[idx] = dot / (xNorm * yNorm);
            }
        }

        const maxIdx = scores.indexOf(Math.max(...scores));
        return [[scores[maxIdx], maxIdx - nPad]];
    }

    // =====================================================
    // LEIDEN CLUSTERING ALGORITHM
    // =====================================================

    /**
     * Leiden clustering algorithm for community detection
     * @param {Object} affinityMat - Sparse affinity matrix
     * @param {number} nSeeds - Number of random seeds to try
     * @param {number} nLeidenIterations - Number of iterations (-1 for convergence)
     * @returns {Array<number>} - Cluster assignments
     */
    leidenCluster(affinityMat, nSeeds = 2, nLeidenIterations = -1) {
        const nVertices = affinityMat.shape[0];
        
        // Create adjacency structure from sparse matrix
        const adjacency = this._sparseMatrixToAdjacency(affinityMat);
        
        let bestClustering = null;
        let bestQuality = null;

        for (let seed = 1; seed <= nSeeds; seed++) {
            const clustering = this._leidenIteration(
                adjacency, affinityMat, nLeidenIterations, seed * 100
            );
            
            const quality = this._calculateModularity(adjacency, clustering, affinityMat);
            
            if (bestQuality === null || quality > bestQuality) {
                bestQuality = quality;
                bestClustering = clustering.slice();
            }
        }

        return bestClustering;
    }

    /**
     * Convert sparse matrix to adjacency list representation
     * @private
     */
    _sparseMatrixToAdjacency(sparseMatrix) {
        const { data, rows, cols, shape } = sparseMatrix;
        const nVertices = shape[0];
        const adjacency = Array(nVertices).fill().map(() => []);
        
        for (let i = 0; i < data.length; i++) {
            const row = rows[i];
            const col = cols[i];
            const weight = data[i];
            
            if (row !== col && weight > 0) { // Skip self-loops and zero weights
                adjacency[row].push({ neighbor: col, weight });
                // Ensure symmetry if not already present
                if (!adjacency[col].some(edge => edge.neighbor === row)) {
                    adjacency[col].push({ neighbor: row, weight });
                }
            }
        }
        
        return adjacency;
    }

    /**
     * Single Leiden iteration
     * @private
     */
    _leidenIteration(adjacency, affinityMat, maxIterations, seed) {
        const nVertices = adjacency.length;
        const rng = this._seedRandom(seed);
        
        // Initialize each vertex in its own community
        let membership = Array.from({length: nVertices}, (_, i) => i);
        
        let iteration = 0;
        let improved = true;
        
        while (improved && (maxIterations === -1 || iteration < maxIterations)) {
            improved = false;
            
            // Shuffle vertex order for unbiased processing
            const vertices = Array.from({length: nVertices}, (_, i) => i);
            this._shuffleArray(vertices, rng);
            
            for (const vertex of vertices) {
                const currentCommunity = membership[vertex];
                let bestCommunity = currentCommunity;
                let bestGain = 0;
                
                // Consider neighboring communities
                const neighborCommunities = new Set();
                for (const { neighbor } of adjacency[vertex]) {
                    neighborCommunities.add(membership[neighbor]);
                }
                
                // Also consider the vertex's current community
                neighborCommunities.add(currentCommunity);
                
                for (const community of neighborCommunities) {
                    if (community === currentCommunity) continue;
                    
                    const gain = this._calculateModularityGain(
                        vertex, currentCommunity, community, membership, adjacency, affinityMat
                    );
                    
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestCommunity = community;
                    }
                }
                
                if (bestCommunity !== currentCommunity) {
                    membership[vertex] = bestCommunity;
                    improved = true;
                }
            }
            
            iteration++;
        }
        
        // Relabel communities to be consecutive integers starting from 0
        return this._relabelCommunities(membership);
    }

    /**
     * Calculate modularity gain for moving vertex to new community
     * @private
     */
    _calculateModularityGain(vertex, oldCommunity, newCommunity, membership, adjacency, affinityMat) {
        if (oldCommunity === newCommunity) return 0;
        
        let deltaQ = 0;
        const totalWeight = this._getTotalWeight(affinityMat);
        
        // Calculate degree of vertex
        const vertexDegree = adjacency[vertex].reduce((sum, edge) => sum + edge.weight, 0);
        
        // Calculate weight of edges from vertex to old community
        let oldCommunityWeight = 0;
        for (const { neighbor, weight } of adjacency[vertex]) {
            if (membership[neighbor] === oldCommunity && neighbor !== vertex) {
                oldCommunityWeight += weight;
            }
        }
        
        // Calculate weight of edges from vertex to new community
        let newCommunityWeight = 0;
        for (const { neighbor, weight } of adjacency[vertex]) {
            if (membership[neighbor] === newCommunity) {
                newCommunityWeight += weight;
            }
        }
        
        // Calculate community degrees (excluding the vertex being moved)
        const oldCommunityDegree = this._getCommunityDegree(oldCommunity, membership, adjacency) - vertexDegree;
        const newCommunityDegree = this._getCommunityDegree(newCommunity, membership, adjacency);
        
        // Modularity gain calculation
        deltaQ = (newCommunityWeight - oldCommunityWeight) / totalWeight;
        deltaQ -= (vertexDegree * (newCommunityDegree - oldCommunityDegree)) / (2 * totalWeight * totalWeight);
        
        return deltaQ;
    }

    /**
     * Get total weight of the graph
     * @private
     */
    _getTotalWeight(affinityMat) {
        return affinityMat.data.reduce((sum, weight) => sum + Math.abs(weight), 0);
    }

    /**
     * Get total degree of a community
     * @private
     */
    _getCommunityDegree(community, membership, adjacency) {
        let degree = 0;
        for (let i = 0; i < membership.length; i++) {
            if (membership[i] === community) {
                degree += adjacency[i].reduce((sum, edge) => sum + edge.weight, 0);
            }
        }
        return degree;
    }

    /**
     * Calculate modularity of a clustering
     * @private
     */
    _calculateModularity(adjacency, membership, affinityMat) {
        const totalWeight = this._getTotalWeight(affinityMat);
        if (totalWeight === 0) return 0;
        
        let modularity = 0;
        const nVertices = adjacency.length;
        
        for (let i = 0; i < nVertices; i++) {
            const degreeI = adjacency[i].reduce((sum, edge) => sum + edge.weight, 0);
            
            for (let j = i; j < nVertices; j++) {
                const degreeJ = adjacency[j].reduce((sum, edge) => sum + edge.weight, 0);
                
                // Find edge weight between i and j
                let edgeWeight = 0;
                for (const { neighbor, weight } of adjacency[i]) {
                    if (neighbor === j) {
                        edgeWeight = weight;
                        break;
                    }
                }
                
                if (membership[i] === membership[j]) {
                    const expectedWeight = (degreeI * degreeJ) / (2 * totalWeight);
                    modularity += edgeWeight - expectedWeight;
                }
            }
        }
        
        return modularity / totalWeight;
    }

    /**
     * Relabel communities to consecutive integers
     * @private
     */
    _relabelCommunities(membership) {
        const uniqueCommunities = [...new Set(membership)].sort((a, b) => a - b);
        const communityMap = new Map();
        
        uniqueCommunities.forEach((community, index) => {
            communityMap.set(community, index);
        });
        
        return membership.map(community => communityMap.get(community));
    }

    /**
     * Shuffle array in place
     * @private
     */
    _shuffleArray(array, rng) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    /**
     * Density adaptation for t-SNE-like clustering (NNTsneConditionalProbs)
     * @param {Array<Array<number>>} affinityMat - Nearest neighbor affinity matrix
     * @param {Array<Array<number>>} nearestNeighbors - Nearest neighbor indices
     * @param {number} perplexity - Target perplexity
     * @returns {Object} - Sparse conditional probability matrix
     */
    densityAdaptation(affinityMat, nearestNeighbors, perplexity) {
        const eps = 1e-8;
        const n = affinityMat.length;

        // Convert to distance matrix (log transform)
        const distMatNN = affinityMat.map(row => 
            row.map(val => Math.log((1.0 / (0.5 * Math.max(val, eps))) - 1))
               .map(val => Math.max(val, 0))
        );

        // Calculate conditional probabilities
        const conditionalP = [];
        for (let i = 0; i < n; i++) {
            const distances = distMatNN[i].slice(1); // Skip self
            const beta = this.binarySearchPerplexity(perplexity, distances);
            
            const probs = distances.map(d => Math.exp(-d * beta));
            const probSum = probs.reduce((sum, p) => sum + p, 0) + eps;
            
            conditionalP.push(probs.map(p => p / probSum));
        }

        // Create sparse matrix representation
        const data = [];
        const rows = [];
        const cols = [];

        for (let i = 0; i < n; i++) {
            const neighbors = nearestNeighbors[i].slice(1); // Skip self
            for (let j = 0; j < neighbors.length; j++) {
                if (conditionalP[i][j] > eps) {
                    data.push(conditionalP[i][j]);
                    rows.push(i);
                    cols.push(neighbors[j]);
                }
            }
        }

        return this.createSparseMatrix(data, rows, cols, n, n);
    }

    // =====================================================
    // MAIN TFMODISCO ALGORITHM (Placeholder)
    // =====================================================

    /**
     * Main TF-MoDISco algorithm
     * @param {Array} oneHot - One-hot encoded sequences [samples x positions x 4]
     * @param {Array} hypotheticalContribs - Attribution scores [samples x positions x 4]
     * @param {Object} options - Algorithm options
     * @returns {Object} - {posPatterns, negPatterns}
     */
    tfmodisco(oneHot, hypotheticalContribs, options = {}) {
        const defaults = {
            slidingWindowSize: 21,
            flankSize: 10,
            minMetaclusterSize: 100,
            weakThresholdForCountingSign: 0.8,
            maxSeqletsPerMetacluster: 20000,
            targetSeqletFDR: 0.2,
            minPassingWindowsFrac: 0.03,
            maxPassingWindowsFrac: 0.2,
            nLeidenRuns: 50,
            nLeidenIterations: -1,
            minOverlapWhileSliding: 0.7,
            nearestNeighborsToCompute: 500,
            affmatCorrelationThreshold: 0.15,
            tsnePerplexity: 10.0,
            fracSupportToTrimTo: 0.2,
            minNumToTrimTo: 30,
            trimToWindowSize: 30,
            initialFlankToAdd: 10,
            finalFlankToAdd: 0,
            subclusterPerplexity: 50,
            finalMinClusterSize: 20,
            minICInWindow: 0.6,
            minICWindowSize: 6,
            ppmPseudocount: 0.001,
            verbose: false
        };

        const params = { ...defaults, ...options };

        if (params.verbose) {
            console.log('Starting TF-MoDISco analysis...');
        }

        // TODO: Implement the full algorithm
        // This is a placeholder structure that will be filled in

        try {
            // 1. Calculate contribution scores
            const contribScores = this._calculateContribScores(oneHot, hypotheticalContribs);

            // 2. Create TrackSet
            const trackSet = new TFModiscoLite.TrackSet(oneHot, contribScores, hypotheticalContribs);

            // 3. Extract seqlets
            const { seqlets, threshold } = this._extractSeqlets(trackSet, params);

            // 4. Split into positive and negative seqlets
            const { posSeqlets, negSeqlets } = this._splitSeqletsBySign(seqlets, params, threshold);

            // 5. Find patterns for positive seqlets
            let posPatterns = null;
            if (posSeqlets.length > params.minMetaclusterSize) {
                posPatterns = this._seqletsToPatterns(posSeqlets, trackSet, 1, params);
            }

            // 6. Find patterns for negative seqlets
            let negPatterns = null;
            if (negSeqlets.length > params.minMetaclusterSize) {
                negPatterns = this._seqletsToPatterns(negSeqlets, trackSet, -1, params);
            }

            return {
                posPatterns: posPatterns || [],
                negPatterns: negPatterns || []
            };

        } catch (error) {
            throw new Error(`TF-MoDISco analysis failed: ${error.message}`);
        }
    }

    /**
     * Calculate contribution scores from one-hot and hypothetical contributions
     * @private
     */
    _calculateContribScores(oneHot, hypotheticalContribs) {
        const contribScores = [];
        
        for (let i = 0; i < oneHot.length; i++) {
            const sampleContrib = [];
            for (let j = 0; j < oneHot[i].length; j++) {
                const posContrib = [];
                for (let k = 0; k < 4; k++) {
                    posContrib.push(oneHot[i][j][k] * hypotheticalContribs[i][j][k]);
                }
                sampleContrib.push(posContrib);
            }
            contribScores.push(sampleContrib);
        }
        
        return contribScores;
    }

    /**
     * Extract seqlets from sequences
     * @private
     */
    _extractSeqlets(trackSet, params) {
        if (params.verbose) {
            console.log('Extracting seqlets...');
        }

        // Sum attribution scores across bases
        const attributionScores = trackSet.contribScores.map(sample => 
            sample.map(pos => pos.reduce((sum, val) => sum + val, 0))
        );

        const { seqlets, threshold } = this.extractSeqlets(
            attributionScores,
            params.slidingWindowSize,
            params.flankSize,
            Math.floor(0.5 * params.slidingWindowSize) + params.flankSize,
            params.targetSeqletFDR,
            params.minPassingWindowsFrac,
            params.maxPassingWindowsFrac,
            params.weakThresholdForCountingSign
        );

        return { 
            seqlets: trackSet.createSeqlets(seqlets),
            threshold 
        };
    }

    /**
     * Main seqlet extraction algorithm
     * @param {Array<Array<number>>} attributionScores - Attribution scores [samples x positions]
     * @param {number} windowSize - Sliding window size
     * @param {number} flank - Flank size
     * @param {number} suppress - Suppression distance
     * @param {number} targetFDR - Target false discovery rate
     * @param {number} minPassingWindowsFrac - Minimum fraction of windows passing threshold
     * @param {number} maxPassingWindowsFrac - Maximum fraction of windows passing threshold
     * @param {number} weakThresholdForCountingSign - Weak threshold for sign counting
     * @returns {Object} - {seqlets, threshold}
     */
    extractSeqlets(attributionScores, windowSize, flank, suppress, targetFDR, 
                   minPassingWindowsFrac, maxPassingWindowsFrac, weakThresholdForCountingSign) {
        
        // 1. Smooth and split attribution scores
        const { posValues, negValues, smoothedTracks } = this._smoothAndSplit(
            attributionScores, windowSize
        );

        // 2. Generate null distributions
        const { posNullValues, negNullValues } = this.laplacianNull(
            smoothedTracks, windowSize, 10000
        );

        // 3. Calculate thresholds using isotonic regression
        let posThreshold = this.isotonicThreshold(
            posValues, posNullValues, true, targetFDR
        );
        let negThreshold = this.isotonicThreshold(
            negValues, negNullValues, false, targetFDR
        );

        // 4. Refine thresholds based on window fractions
        const allValues = [...posValues, ...negValues];
        ({ posThreshold, negThreshold } = this._refineThresholds(
            allValues, posThreshold, negThreshold, 
            minPassingWindowsFrac, maxPassingWindowsFrac
        ));

        // 5. Create distribution for weak threshold calculation
        const distribution = allValues.map(Math.abs).sort((a, b) => a - b);
        
        const transformedPosThreshold = Math.sign(posThreshold) * 
            this._searchSorted(distribution, Math.abs(posThreshold)) / distribution.length;
        const transformedNegThreshold = Math.sign(negThreshold) * 
            this._searchSorted(distribution, Math.abs(negThreshold)) / distribution.length;

        // 6. Apply thresholds and suppress flanks
        const thresholdedTracks = smoothedTracks.map(track => {
            const result = track.map(val => {
                if (val >= posThreshold || val <= negThreshold) {
                    return Math.abs(val);
                } else {
                    return -Infinity;
                }
            });
            
            // Suppress flanks
            for (let i = 0; i < flank; i++) {
                result[i] = -Infinity;
                result[result.length - 1 - i] = -Infinity;
            }
            
            return result;
        });

        // 7. Extract seqlets iteratively
        const seqlets = this._iterativeExtractSeqlets(
            thresholdedTracks, windowSize, flank, suppress
        );

        // 8. Calculate weak threshold
        const weakThresh = Math.min(
            Math.min(transformedPosThreshold, Math.abs(transformedNegThreshold)) - 0.0001,
            weakThresholdForCountingSign
        );
        const threshold = distribution[Math.floor(weakThresh * distribution.length)];

        return { seqlets, threshold };
    }

    /**
     * Smooth attribution scores and split into positive/negative
     * @private
     */
    _smoothAndSplit(tracks, windowSize, subsampleCap = 1000000) {
        const n = tracks.length;
        
        // Convert to cumulative sum for efficient sliding window
        const cumulativeTracks = tracks.map(track => {
            const padded = [0, ...this.cumulativeSum(track)];
            const smoothed = [];
            for (let i = windowSize; i < padded.length; i++) {
                smoothed.push(padded[i] - padded[i - windowSize]);
            }
            return smoothed;
        });

        // Flatten and subsample if needed
        let values = cumulativeTracks.flat();
        if (values.length > subsampleCap) {
            const rng = this._seedRandom(1234);
            const shuffled = values.slice();
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            values = shuffled.slice(0, subsampleCap);
        }

        const posValues = values.filter(v => v >= 0).sort((a, b) => a - b);
        const negValues = values.filter(v => v < 0).sort((a, b) => b - a);

        return { posValues, negValues, smoothedTracks: cumulativeTracks };
    }

    /**
     * Refine thresholds based on window passing fractions
     * @private
     */
    _refineThresholds(vals, posThreshold, negThreshold, minPassingWindowsFrac, maxPassingWindowsFrac) {
        const fracPassingWindows = (vals.filter(v => v >= posThreshold).length + 
                                   vals.filter(v => v <= negThreshold).length) / vals.length;

        if (fracPassingWindows < minPassingWindowsFrac) {
            const absVals = vals.map(Math.abs).sort((a, b) => b - a);
            const threshold = absVals[Math.floor((1 - minPassingWindowsFrac) * absVals.length)];
            posThreshold = threshold;
            negThreshold = -threshold;
        }

        if (fracPassingWindows > maxPassingWindowsFrac) {
            const absVals = vals.map(Math.abs).sort((a, b) => b - a);
            const threshold = absVals[Math.floor((1 - maxPassingWindowsFrac) * absVals.length)];
            posThreshold = threshold;
            negThreshold = -threshold;
        }

        return { posThreshold, negThreshold };
    }

    /**
     * Binary search to find insertion point (like numpy searchsorted)
     * @private
     */
    _searchSorted(sortedArray, value) {
        let left = 0;
        let right = sortedArray.length;
        
        while (left < right) {
            const mid = Math.floor((left + right) / 2);
            if (sortedArray[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }

    /**
     * Iteratively extract seqlets with suppression
     * @private
     */
    _iterativeExtractSeqlets(scoreTracks, windowSize, flank, suppress) {
        const seqlets = [];
        
        for (let exampleIdx = 0; exampleIdx < scoreTracks.length; exampleIdx++) {
            const track = scoreTracks[exampleIdx].slice(); // Copy
            const trackLength = track.length;
            
            while (true) {
                // Find maximum
                const maxIdx = track.indexOf(Math.max(...track));
                const maxVal = track[maxIdx];
                
                // Break if no more peaks above threshold
                if (maxVal === -Infinity) {
                    break;
                }
                
                // Check if we can expand without going off edge
                if (maxIdx >= flank && maxIdx < trackLength - flank) {
                    const seqlet = new TFModiscoLite.Seqlet(
                        exampleIdx,
                        maxIdx - flank,
                        maxIdx + windowSize + flank,
                        false
                    );
                    seqlets.push(seqlet);
                }
                
                // Suppress surrounding region
                const leftIdx = Math.max(Math.floor(maxIdx - suppress + 0.5), 0);
                const rightIdx = Math.min(Math.ceil(maxIdx + suppress + 0.5), trackLength);
                
                for (let i = leftIdx; i < rightIdx; i++) {
                    track[i] = -Infinity;
                }
            }
        }
        
        return seqlets;
    }

    /**
     * Split seqlets by attribution sign
     * @private
     */
    _splitSeqletsBySign(seqlets, params, threshold) {
        const posSeqlets = [];
        const negSeqlets = [];

        for (const seqlet of seqlets) {
            // Calculate core attribution sum (excluding flanks)
            const coreFlank = Math.floor(0.5 * (seqlet.length - params.slidingWindowSize));
            const coreStart = coreFlank;
            const coreEnd = seqlet.contribScores.length - coreFlank;
            
            let attrSum = 0;
            for (let i = coreStart; i < coreEnd; i++) {
                for (let j = 0; j < 4; j++) {
                    attrSum += seqlet.contribScores[i][j];
                }
            }

            if (attrSum > threshold) {
                posSeqlets.push(seqlet);
            } else if (attrSum < -threshold) {
                negSeqlets.push(seqlet);
            }
        }

        return { posSeqlets, negSeqlets };
    }

    /**
     * Convert seqlets to patterns (placeholder)
     * @private
     */
    _seqletsToPatterns(seqlets, trackSet, trackSign, params) {
        if (params.verbose) {
            console.log(`Finding patterns for ${trackSign > 0 ? 'positive' : 'negative'} seqlets (${seqlets.length} seqlets)...`);
        }

        if (seqlets.length < params.minMetaclusterSize) {
            if (params.verbose) {
                console.log(`Insufficient seqlets (${seqlets.length} < ${params.minMetaclusterSize})`);
            }
            return [];
        }

        // 1. Subsample seqlets if too many
        let workingSeqlets = seqlets;
        if (seqlets.length > params.maxSeqletsPerMetacluster) {
            workingSeqlets = this._subsampleSeqlets(seqlets, params.maxSeqletsPerMetacluster);
            if (params.verbose) {
                console.log(`Subsampled to ${workingSeqlets.length} seqlets`);
            }
        }

        // 2. Calculate similarity matrices
        if (params.verbose) {
            console.log('Computing similarity matrices...');
        }
        
        const cosineMatrix = this.computeCosineMatrix(workingSeqlets, params.nearestNeighborsToCompute);
        const jaccardMatrix = this.computeJaccardMatrix(workingSeqlets, params.nearestNeighborsToCompute);

        // 3. Create combined affinity matrix
        const affinityMatrix = this._combineAffinityMatrices(
            cosineMatrix, jaccardMatrix, params.affmatCorrelationThreshold
        );

        // 4. Perform clustering
        if (params.verbose) {
            console.log('Clustering seqlets...');
        }
        
        const clusters = this._clusterSeqlets(affinityMatrix, workingSeqlets, params);

        // 5. Form initial patterns from clusters
        if (params.verbose) {
            console.log(`Found ${clusters.length} initial clusters`);
        }
        
        let patterns = this._formPatternsFromClusters(clusters, trackSet, params);

        // 6. Merge similar patterns
        if (patterns.length > 1) {
            if (params.verbose) {
                console.log('Merging similar patterns...');
            }
            patterns = this._mergeSimilarPatterns(patterns, params);
        }

        // 7. Final pattern refinement
        patterns = this._refinePatterns(patterns, params);

        if (params.verbose) {
            console.log(`Final ${patterns.length} patterns for ${trackSign > 0 ? 'positive' : 'negative'} seqlets`);
        }

        return patterns;
    }

    /**
     * Subsample seqlets randomly
     * @private
     */
    _subsampleSeqlets(seqlets, maxCount) {
        const indices = Array.from({length: seqlets.length}, (_, i) => i);
        
        // Fisher-Yates shuffle
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        return indices.slice(0, maxCount).map(i => seqlets[i]);
    }

    /**
     * Combine cosine and Jaccard affinity matrices
     * @private
     */
    _combineAffinityMatrices(cosineMatrix, jaccardMatrix, correlationThreshold) {
        const n = cosineMatrix.length;
        const combinedMatrix = [];

        for (let i = 0; i < n; i++) {
            const row = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    row.push(1.0);
                } else {
                    // Combine similarities with weighted average
                    const cosine = cosineMatrix[i][j] || 0;
                    const jaccard = jaccardMatrix[i][j] || 0;
                    
                    // Weight more toward cosine if above correlation threshold
                    const weight = Math.max(cosine, jaccard) > correlationThreshold ? 0.7 : 0.5;
                    const combined = weight * cosine + (1 - weight) * jaccard;
                    
                    row.push(Math.max(0, combined));
                }
            }
            combinedMatrix.push(row);
        }

        return combinedMatrix;
    }

    /**
     * Cluster seqlets using Leiden algorithm
     * @private
     */
    _clusterSeqlets(affinityMatrix, seqlets, params) {
        // Apply density adaptation for t-SNE-like clustering
        const nearestNeighbors = this._findNearestNeighbors(affinityMatrix, params.nearestNeighborsToCompute);
        const adaptedMatrix = this.densityAdaptation(affinityMatrix, nearestNeighbors, params.tsnePerplexity);

        // Run Leiden clustering
        const assignments = this.leidenCluster(
            adaptedMatrix, 
            params.nLeidenRuns, 
            params.nLeidenIterations
        );

        // Group seqlets by cluster assignment
        const clusters = {};
        for (let i = 0; i < assignments.length; i++) {
            const clusterId = assignments[i];
            if (!clusters[clusterId]) {
                clusters[clusterId] = [];
            }
            clusters[clusterId].push(seqlets[i]);
        }

        // Filter out clusters that are too small
        return Object.values(clusters).filter(cluster => cluster.length >= params.finalMinClusterSize);
    }

    /**
     * Find nearest neighbors for each point
     * @private
     */
    _findNearestNeighbors(affinityMatrix, k) {
        const n = affinityMatrix.length;
        const neighbors = [];

        for (let i = 0; i < n; i++) {
            const distances = affinityMatrix[i]
                .map((dist, idx) => ({ dist: 1 - dist, idx })) // Convert similarity to distance
                .sort((a, b) => a.dist - b.dist)
                .slice(0, Math.min(k, n))
                .map(item => item.idx);
            
            neighbors.push(distances);
        }

        return neighbors;
    }

    /**
     * Form patterns from clusters of seqlets
     * @private
     */
    _formPatternsFromClusters(clusters, trackSet, params) {
        const patterns = [];

        for (let i = 0; i < clusters.length; i++) {
            const cluster = clusters[i];
            
            if (cluster.length < params.finalMinClusterSize) {
                continue;
            }

            // Create SeqletSet from cluster
            const seqletSet = new TFModiscoLite.SeqletSet(cluster);
            
            // Calculate pattern representations
            this._calculatePatternRepresentations(seqletSet, trackSet, params);
            
            // Check if pattern meets information content requirements
            if (this._passesICFilter(seqletSet, params)) {
                patterns.push({
                    seqlets: cluster,
                    seqletSet: seqletSet,
                    size: cluster.length,
                    id: `pattern_${i}`
                });
            }
        }

        return patterns;
    }

    /**
     * Calculate pattern representations (PWM, contribution matrix, etc.)
     * @private
     */
    _calculatePatternRepresentations(seqletSet, trackSet, params) {
        const seqlets = seqletSet.seqlets;
        const seqLen = seqlets[0].sequence.length;
        
        // Initialize matrices
        const sequenceMatrix = Array(4).fill().map(() => Array(seqLen).fill(0));
        const contribMatrix = Array(4).fill().map(() => Array(seqLen).fill(0));
        const hypotheticalMatrix = Array(4).fill().map(() => Array(seqLen).fill(0));

        // Accumulate across all seqlets
        for (const seqlet of seqlets) {
            for (let pos = 0; pos < seqLen; pos++) {
                for (let base = 0; base < 4; base++) {
                    sequenceMatrix[base][pos] += seqlet.sequence[pos][base];
                    contribMatrix[base][pos] += seqlet.contribScores[pos][base];
                    hypotheticalMatrix[base][pos] += seqlet.hypotheticalContribs[pos][base];
                }
            }
        }

        // Normalize to create PWM
        for (let pos = 0; pos < seqLen; pos++) {
            const total = sequenceMatrix.reduce((sum, row) => sum + row[pos], 0);
            if (total > 0) {
                for (let base = 0; base < 4; base++) {
                    sequenceMatrix[base][pos] = (sequenceMatrix[base][pos] + params.ppmPseudocount) / 
                                               (total + 4 * params.ppmPseudocount);
                }
            }
        }

        seqletSet.sequence = sequenceMatrix;
        seqletSet.contribScores = contribMatrix;
        seqletSet.hypotheticalContribs = hypotheticalMatrix;
    }

    /**
     * Check if pattern passes information content filter
     * @private
     */
    _passesICFilter(seqletSet, params) {
        const pwm = seqletSet.sequence;
        const icProfile = this.computeIC(pwm);
        
        // Find windows that pass IC threshold
        const windowSize = params.minICWindowSize;
        const passingWindows = [];
        
        for (let i = 0; i <= icProfile.length - windowSize; i++) {
            const windowIC = icProfile.slice(i, i + windowSize).reduce((sum, ic) => sum + ic, 0) / windowSize;
            if (windowIC >= params.minICInWindow) {
                passingWindows.push(i);
            }
        }

        return passingWindows.length > 0;
    }

    /**
     * Merge similar patterns
     * @private
     */
    _mergeSimilarPatterns(patterns, params) {
        if (patterns.length <= 1) {
            return patterns;
        }

        const mergeThreshold = 0.8; // High similarity threshold for merging
        const merged = [];
        const used = new Set();

        for (let i = 0; i < patterns.length; i++) {
            if (used.has(i)) continue;

            const basePattern = patterns[i];
            const toMerge = [basePattern];
            used.add(i);

            // Find similar patterns to merge
            for (let j = i + 1; j < patterns.length; j++) {
                if (used.has(j)) continue;

                const similarity = this._computePatternSimilarity(basePattern.seqletSet, patterns[j].seqletSet);
                if (similarity > mergeThreshold) {
                    toMerge.push(patterns[j]);
                    used.add(j);
                }
            }

            // Merge patterns if multiple found
            if (toMerge.length > 1) {
                const mergedPattern = this._mergePatterns(toMerge, params);
                merged.push(mergedPattern);
            } else {
                merged.push(basePattern);
            }
        }

        return merged;
    }

    /**
     * Compute similarity between two patterns
     * @private
     */
    _computePatternSimilarity(pattern1, pattern2) {
        // Simple cosine similarity between PWMs
        const pwm1 = pattern1.sequence;
        const pwm2 = pattern2.sequence;

        if (pwm1[0].length !== pwm2[0].length) {
            return 0; // Different lengths
        }

        // Flatten PWMs
        const vec1 = pwm1.flat();
        const vec2 = pwm2.flat();

        // Compute cosine similarity
        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

        return dotProduct / (norm1 * norm2);
    }

    /**
     * Merge multiple patterns into one
     * @private
     */
    _mergePatterns(patterns, params) {
        // Combine all seqlets
        const allSeqlets = patterns.flatMap(p => p.seqlets);
        
        // Create new seqlet set
        const mergedSeqletSet = new TFModiscoLite.SeqletSet(allSeqlets);
        
        // Recalculate representations
        this._calculatePatternRepresentations(mergedSeqletSet, null, params);
        
        return {
            seqlets: allSeqlets,
            seqletSet: mergedSeqletSet,
            size: allSeqlets.length,
            id: `merged_${patterns.map(p => p.id).join('_')}`
        };
    }

    /**
     * Final pattern refinement
     * @private
     */
    _refinePatterns(patterns, params) {
        // Sort by size (largest first)
        patterns.sort((a, b) => b.size - a.size);
        
        // Trim patterns to focus on high-information regions
        for (const pattern of patterns) {
            this._trimPattern(pattern, params);
        }

        return patterns;
    }

    /**
     * Trim pattern to focus on high-information content region
     * @private
     */
    _trimPattern(pattern, params) {
        const pwm = pattern.seqletSet.sequence;
        const icProfile = this.computeIC(pwm);
        
        // Find the region with highest average information content
        const windowSize = params.trimToWindowSize;
        let bestStart = 0;
        let bestIC = 0;
        
        for (let i = 0; i <= icProfile.length - windowSize; i++) {
            const windowIC = icProfile.slice(i, i + windowSize).reduce((sum, ic) => sum + ic, 0);
            if (windowIC > bestIC) {
                bestIC = windowIC;
                bestStart = i;
            }
        }
        
        // Update pattern to focus on this region (this is a simplified version)
        // In practice, you'd want to re-extract seqlets focused on this core region
        pattern.coreStart = bestStart;
        pattern.coreEnd = bestStart + windowSize;
    }

}

// =====================================================
// EXPORT FOR BOTH NODE.JS AND BROWSER
// =====================================================

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TFModiscoLite;
} else if (typeof window !== 'undefined') {
    window.TFModiscoLite = TFModiscoLite;
}