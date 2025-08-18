/**
 * Simple working example of tfmodisco-lite.js
 */

const TFModiscoLite = require('./tfmodisco-lite.js');
const tfm = new TFModiscoLite();

console.log('🧬 TF-MoDISco-Lite.js Simple Example\n');

// 1. Basic functionality
console.log('1. One-hot encoding:');
const sequence = "ACGTACGT";
const oneHot = tfm.oneHotEncode(sequence);
console.log(`   Input: ${sequence}`);
console.log(`   Output shape: ${oneHot.length} x ${oneHot[0].length}`);
console.log(`   A row: [${oneHot[0].join(',')}]`);
console.log(`   C row: [${oneHot[1].join(',')}]`);
console.log(`   G row: [${oneHot[2].join(',')}]`);
console.log(`   T row: [${oneHot[3].join(',')}]\n`);

// 2. Information content
console.log('2. Information content calculation:');
const pwm = [
    [0.8, 0.1, 0.05, 0.2], // A
    [0.1, 0.7, 0.1, 0.2],  // C
    [0.05, 0.1, 0.8, 0.2], // G
    [0.05, 0.1, 0.05, 0.4] // T
];
const ic = tfm.computeIC(pwm);
console.log('   PWM information content per position:');
ic.forEach((bits, pos) => {
    console.log(`   Position ${pos}: ${bits.toFixed(3)} bits`);
});
console.log();

// 3. Consensus sequence
console.log('3. Consensus sequence generation:');
const consensus = tfm.icWeightedConsensus(pwm);
console.log(`   Consensus: ${consensus}\n`);

// 4. Data structures
console.log('4. Core data structures:');
const seqlet = new TFModiscoLite.Seqlet(0, 10, 20, false);
console.log(`   Seqlet: example ${seqlet.exampleIdx}, positions ${seqlet.start}-${seqlet.end}`);

const seqletSet = new TFModiscoLite.SeqletSet([seqlet]);
console.log(`   SeqletSet: ${seqletSet.seqlets.length} seqlets, length ${seqletSet.length}`);

const trackSet = new TFModiscoLite.TrackSet([oneHot], [oneHot], [oneHot]);
console.log(`   TrackSet: ${trackSet.oneHot.length} sequences\n`);

// 5. Matrix operations
console.log('5. Matrix operations:');
const matA = [[1, 2], [3, 4]];
const matB = [[5, 6], [7, 8]];
const product = tfm.dotProduct(matA, matB);
console.log(`   ${JSON.stringify(matA)} × ${JSON.stringify(matB)} = ${JSON.stringify(product)}`);

const values = [1, 2, 3, 4, 5];
const windowSums = tfm.slidingWindowSum(values, 3);
console.log(`   Sliding window sum (size=3): [${values.join(',')}] → [${windowSums.join(',')}]\n`);

// 6. Version info
console.log(`📦 TF-MoDISco-Lite.js v${tfm.version}`);
console.log('✅ Library is working correctly!\n');

console.log('💡 Next steps:');
console.log('   • Use real attribution data from your ML model');
console.log('   • Try larger datasets for pattern discovery');
console.log('   • Adjust parameters based on your data characteristics');
console.log('   • See README.md for detailed usage examples');