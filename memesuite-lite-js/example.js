/**
 * Example usage of memesuite-lite.js
 */

// Import the library (Node.js)
const MemeSuiteLite = require('./memesuite-lite.js');

// Create instance
const msl = new MemeSuiteLite();

console.log('='.repeat(60));
console.log('MemeSuite-Lite.js Example');
console.log('='.repeat(60));

// Example 1: Basic one-hot encoding
console.log('\n1. One-hot encoding example:');
const sequence = "ACGTACGT";
const oneHot = msl.oneHotEncode(sequence);
console.log(`Sequence: ${sequence}`);
console.log('One-hot encoding:');
oneHot.forEach((row, i) => {
    console.log(`  ${'ACGT'[i]}: [${row.join(',')}]`);
});

// Convert back
const backToSeq = msl.characters(oneHot);
console.log(`Back to sequence: ${backToSeq}`);

// Example 2: Working with motifs
console.log('\n2. MEME file parsing example:');
const sampleMeme = `MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF SAMPLE_TF
letter-probability matrix: alength= 4 w= 4 nsites= 1 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.05 0.1 0.8 0.05
0.1 0.1 0.1 0.7
URL BLANK

MOTIF ANOTHER_TF
letter-probability matrix: alength= 4 w= 3 nsites= 1 E= 0
0.9 0.0 0.1
0.0 0.9 0.0
0.05 0.05 0.8
0.05 0.05 0.1
URL BLANK
`;

const motifs = msl.readMeme(sampleMeme);
console.log(`Parsed ${Object.keys(motifs).length} motifs:`);
Object.keys(motifs).forEach(name => {
    const pwm = motifs[name];
    const consensus = msl.characters(pwm, ['A','C','G','T'], true);
    console.log(`  ${name}: ${consensus} (length: ${pwm[0].length})`);
});

// Example 3: FIMO analysis
console.log('\n3. FIMO analysis example:');
const testSequences = [
    "ACGTACGTACGTACGT",
    "TTTTAAAACCCCGGGG",
    "GCTAGCTAGCTAGCTA",
    "ACGTNNNNACGT"
];

console.log('Test sequences:');
testSequences.forEach((seq, i) => {
    console.log(`  Seq${i+1}: ${seq}`);
});

const fimoResults = msl.runFimo(sampleMeme, testSequences, {
    threshold: 0.1,
    reverseComplement: true
});

console.log('\nFIMO Results:');
fimoResults.forEach(result => {
    console.log(`\nMotif: ${result.motif_name}`);
    if (result.hits.length === 0) {
        console.log('  No hits found');
    } else {
        result.hits.forEach((hit, i) => {
            console.log(`  Hit ${i+1}: Seq${hit.sequence_idx+1} pos ${hit.start}-${hit.end} ` +
                       `(${hit.strand}) score=${hit.score.toFixed(3)} p=${hit.p_value.toExponential(2)}`);
        });
    }
});

// Example 4: TOMTOM comparison
console.log('\n4. TOMTOM comparison example:');

const queryMeme = `MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF QUERY_MOTIF
letter-probability matrix: alength= 4 w= 4 nsites= 1 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.05 0.1 0.8 0.05
0.1 0.1 0.1 0.7
URL BLANK
`;

const targetMeme = sampleMeme;

const tomtomResults = msl.runTomtom(queryMeme, targetMeme);

console.log('TOMTOM Results:');
console.log('Query vs Target P-values:');
tomtomResults.pValues.forEach((row, i) => {
    row.forEach((pval, j) => {
        console.log(`  Query ${i} vs Target ${j}: p-value = ${pval.toExponential(3)}, ` +
                   `score = ${tomtomResults.scores[i][j].toFixed(2)}, ` +
                   `offset = ${tomtomResults.offsets[i][j]}`);
    });
});

// Example 5: Utility functions
console.log('\n5. Utility functions example:');

// Test with sequences containing N
const nSequence = "ACGTNNNACGT";
try {
    const nOneHot = msl.oneHotEncode(nSequence);
    console.log(`Sequence with N: ${nSequence}`);
    console.log('One-hot (N positions will be all zeros):');
    nOneHot.forEach((row, i) => {
        console.log(`  ${'ACGT'[i]}: [${row.join(',')}]`);
    });
} catch (e) {
    console.log(`Error: ${e.message}`);
}

// Test character conversion with custom alphabet
console.log('\nCustom alphabet example:');
const customPwm = [
    [0.8, 0.1, 0.1],  // A
    [0.1, 0.8, 0.1],  // C  
    [0.05, 0.05, 0.7], // G
    [0.05, 0.05, 0.1], // T
    [0.0, 0.0, 0.0]    // N
];

try {
    const customSeq = msl.characters(customPwm, ['A','C','G','T','N'], true);
    console.log(`Custom PWM consensus: ${customSeq}`);
} catch (e) {
    console.log(`Error: ${e.message}`);
}

console.log('\n' + '='.repeat(60));
console.log('Example completed!');
console.log('='.repeat(60));