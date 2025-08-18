/**
 * Example usage of tfmodisco-lite.js
 */

// Import the library (Node.js)
const TFModiscoLite = require('./tfmodisco-lite.js');

// Create instance
const tfm = new TFModiscoLite();

console.log('='.repeat(70));
console.log('TF-MoDISco-Lite.js Example');
console.log('='.repeat(70));

// Example 1: Basic one-hot encoding
console.log('\n1. One-hot encoding example:');
const sequence = "ACGTACGT";
const oneHot = tfm.oneHotEncode(sequence);
console.log(`Sequence: ${sequence}`);
console.log('One-hot encoding:');
oneHot.forEach((row, i) => {
    console.log(`  ${'ACGT'[i]}: [${row.join(',')}]`);
});

// Convert back to consensus
const consensus = tfm.icWeightedConsensus(oneHot);
console.log(`Consensus sequence: ${consensus}`);

// Example 2: Information content calculation
console.log('\n2. Information content calculation:');
const samplePWM = [
    [0.8, 0.1, 0.05, 0.05], // Strong A preference
    [0.1, 0.7, 0.1, 0.1],   // Strong C preference  
    [0.05, 0.1, 0.8, 0.05], // Strong G preference
    [0.2, 0.2, 0.2, 0.4]    // Weak T preference
];

const icProfile = tfm.computeIC(samplePWM);
console.log('Sample PWM information content:');
icProfile.forEach((ic, pos) => {
    console.log(`  Position ${pos}: ${ic.toFixed(3)} bits`);
});

// Example 3: Generate synthetic data for TF-MoDISco
console.log('\n3. Synthetic data generation for TF-MoDISco:');

function generateSyntheticData() {
    const sequenceLength = 50;
    const numSequences = 20;
    const motifLength = 8;
    
    const sequences = [];
    const attributions = [];
    
    console.log('Generating synthetic sequences with embedded motifs...');
    
    for (let i = 0; i < numSequences; i++) {
        // Generate random background sequence
        const seq = [];
        const attr = [];
        
        for (let pos = 0; pos < sequenceLength; pos++) {
            // Random base with slight bias
            const baseProbs = [0.27, 0.23, 0.23, 0.27]; // A, C, G, T
            let base = 0;
            const rand = Math.random();
            let cumProb = 0;
            for (let b = 0; b < 4; b++) {
                cumProb += baseProbs[b];
                if (rand < cumProb) {
                    base = b;
                    break;
                }
            }
            
            // One-hot encode
            const oneHotPos = [0, 0, 0, 0];
            oneHotPos[base] = 1;
            seq.push(oneHotPos);
            
            // Low background attribution
            attr.push([0.05, 0.05, 0.05, 0.05]);
        }
        
        // Add motif with high attribution (50% chance)
        if (Math.random() < 0.5) {
            const motifStart = Math.floor(Math.random() * (sequenceLength - motifLength));
            
            // Strong GATA motif: WGATAR
            const motifPattern = [
                [0.4, 0.1, 0.1, 0.4], // W (A/T)
                [0.05, 0.05, 0.85, 0.05], // G
                [0.9, 0.03, 0.04, 0.03], // A
                [0.05, 0.05, 0.05, 0.85], // T
                [0.8, 0.1, 0.05, 0.05], // A
                [0.3, 0.1, 0.5, 0.1]  // R (A/G)
            ];
            
            for (let m = 0; m < Math.min(motifPattern.length, motifLength); m++) {
                if (motifStart + m < sequenceLength) {
                    const pos = motifStart + m;
                    
                    // Sample from motif distribution
                    const probs = motifPattern[m];
                    let base = 0;
                    const rand = Math.random();
                    let cumProb = 0;
                    for (let b = 0; b < 4; b++) {
                        cumProb += probs[b];
                        if (rand < cumProb) {
                            base = b;
                            break;
                        }
                    }
                    
                    // Update sequence
                    seq[pos] = [0, 0, 0, 0];
                    seq[pos][base] = 1;
                    
                    // High attribution scores
                    attr[pos] = probs.map(p => p * 2.0); // Scale up attributions
                }
            }
        }
        
        sequences.push(seq);
        attributions.push(attr);
    }
    
    return { sequences, attributions };
}

const { sequences, attributions } = generateSyntheticData();
console.log(`Generated ${sequences.length} sequences of length ${sequences[0].length}`);

// Example 4: Run TF-MoDISco analysis
console.log('\n4. Running TF-MoDISco analysis:');

try {
    const results = tfm.tfmodisco(sequences, attributions, {
        slidingWindowSize: 15,
        flankSize: 5,
        targetSeqletFDR: 0.3,
        minMetaclusterSize: 3,  // Small for demo
        finalMinClusterSize: 2,
        maxSeqletsPerMetacluster: 100,
        verbose: true
    });
    
    console.log('\nTF-MoDISco Results:');
    console.log(`Found ${results.posPatterns.length} positive patterns`);
    console.log(`Found ${results.negPatterns.length} negative patterns`);
    
    // Display positive patterns
    results.posPatterns.forEach((pattern, i) => {
        console.log(`\nPositive Pattern ${i + 1}:`);
        console.log(`  Seqlets: ${pattern.size}`);
        console.log(`  Core region: ${pattern.coreStart || 'N/A'}-${pattern.coreEnd || 'N/A'}`);
        
        if (pattern.seqletSet && pattern.seqletSet.sequence) {
            const pwm = pattern.seqletSet.sequence;
            const consensus = tfm.icWeightedConsensus(pwm);
            const totalIC = tfm.computeIC(pwm).reduce((sum, ic) => sum + ic, 0);
            
            console.log(`  Consensus: ${consensus}`);
            console.log(`  Total IC: ${totalIC.toFixed(2)} bits`);
            console.log('  PWM:');
            
            for (let base = 0; base < 4; base++) {
                const row = pwm[base].map(val => val.toFixed(2)).join(' ');
                console.log(`    ${'ACGT'[base]}: ${row}`);
            }
        }
    });
    
    // Display negative patterns
    results.negPatterns.forEach((pattern, i) => {
        console.log(`\nNegative Pattern ${i + 1}:`);
        console.log(`  Seqlets: ${pattern.size}`);
        
        if (pattern.seqletSet && pattern.seqletSet.sequence) {
            const consensus = tfm.icWeightedConsensus(pattern.seqletSet.sequence);
            console.log(`  Consensus: ${consensus}`);
        }
    });
    
} catch (error) {
    console.error('TF-MoDISco analysis failed:', error.message);
    console.error('This is expected with small synthetic data - try with real data!');
}

// Example 5: Utility functions demonstration
console.log('\n5. Utility functions demonstration:');

// Test k-mer representation
console.log('\nK-mer representation:');
const testSeq = "ACGTACGT";
const testOneHot = tfm.oneHotEncode(testSeq);
const kmerRepr = tfm.gappedKmerRepr([testOneHot], 3, 2);
console.log(`Sequence: ${testSeq}`);
console.log(`3-mer representation (gap=2):`, kmerRepr.nnzData.slice(0, 10));

// Test sliding window operations
console.log('\nSliding window sum:');
const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const windowSums = tfm.slidingWindowSum(values, 3);
console.log(`Values: [${values.join(', ')}]`);
console.log(`Window sums (size=3): [${windowSums.join(', ')}]`);

// Test matrix operations
console.log('\nMatrix operations:');
const matA = [[1, 2], [3, 4]];
const matB = [[5, 6], [7, 8]];
const dotProduct = tfm.dotProduct(matA, matB);
console.log('Matrix A:', matA);
console.log('Matrix B:', matB);
console.log('Dot product:', dotProduct);

console.log('\n' + '='.repeat(70));
console.log('Example completed! Try running with your own sequence data.');
console.log('='.repeat(70));