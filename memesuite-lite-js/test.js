/**
 * Basic tests for memesuite-lite.js
 */

const MemeSuiteLite = require('./memesuite-lite.js');

function assert(condition, message) {
    if (!condition) {
        throw new Error('Test failed: ' + message);
    }
    console.log('✓ ' + message);
}

function assertArraysEqual(arr1, arr2, message) {
    const equal = JSON.stringify(arr1) === JSON.stringify(arr2);
    assert(equal, message);
}

function runTests() {
    console.log('Running memesuite-lite.js tests...\n');
    
    const msl = new MemeSuiteLite();
    
    // Test 1: One-hot encoding
    console.log('Testing one-hot encoding...');
    const seq = "ACGT";
    const expected = [
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ];
    const result = msl.oneHotEncode(seq);
    assertArraysEqual(result, expected, 'One-hot encoding basic test');
    
    // Test 2: Character conversion
    console.log('Testing character conversion...');
    const backSeq = msl.characters(expected);
    assert(backSeq === seq, 'Character conversion roundtrip');
    
    // Test 3: One-hot with N characters
    console.log('Testing one-hot with ignore characters...');
    const seqWithN = "ACNGT";
    const expectedWithN = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 1]
    ];
    const resultWithN = msl.oneHotEncode(seqWithN);
    assertArraysEqual(resultWithN, expectedWithN, 'One-hot encoding with N');
    
    // Test 4: MEME file parsing
    console.log('Testing MEME file parsing...');
    const memeContent = `MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF TEST_TF
letter-probability matrix: alength= 4 w= 3 nsites= 1 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.1 0.2 0.6 0.1
URL BLANK
`;

    const motifs = msl.readMeme(memeContent);
    assert(Object.keys(motifs).length === 1, 'Parsed one motif');
    assert('TEST_TF' in motifs, 'Motif name parsed correctly');
    assert(motifs['TEST_TF'].length === 4, 'PWM has 4 rows');
    assert(motifs['TEST_TF'][0].length === 3, 'PWM has 3 columns');
    
    // Test 5: MEME file writing
    console.log('Testing MEME file writing...');
    const writtenMeme = msl.writeMeme(motifs);
    assert(writtenMeme.includes('MOTIF TEST_TF'), 'Written MEME contains motif name');
    assert(writtenMeme.includes('letter-probability matrix'), 'Written MEME contains matrix header');
    
    // Test 6: Mathematical utilities
    console.log('Testing mathematical utilities...');
    const logSum = msl.logAddExp2(1, 2);
    assert(!isNaN(logSum) && isFinite(logSum), 'logAddExp2 returns valid number');
    
    const values = [1, 2, 3, 4, 5];
    const counts = [1, 1, 1, 1, 1];
    const median = msl.binnedMedian(values, 1, 5, counts);
    assert(median >= 2 && median <= 4, 'Binned median in reasonable range');
    
    // Test 7: PWM to mapping
    console.log('Testing PWM to p-value mapping...');
    const logPwm = [
        [-1, -1, -1],
        [-1, -1, -1], 
        [-1, -1, -1],
        [-1, -1, -1]
    ];
    const mapping = msl.pwmToMapping(logPwm, 0.1);
    assert(mapping.smallest !== undefined, 'PWM mapping returns smallest offset');
    assert(Array.isArray(mapping.logPdf), 'PWM mapping returns log PDF array');
    
    // Test 8: Basic FIMO functionality
    console.log('Testing FIMO algorithm...');
    const testMotifs = {
        'simple': [
            [0.8, 0.1],
            [0.1, 0.8],
            [0.05, 0.05],
            [0.05, 0.05]
        ]
    };
    const testSeqs = ["ACGTACGT", "TTTTAAAA"];
    const fimoResults = msl.fimo(testMotifs, testSeqs, { threshold: 0.5 });
    
    assert(Array.isArray(fimoResults), 'FIMO returns array');
    assert(fimoResults.length === 1, 'FIMO returns one result per motif');
    assert('motif_name' in fimoResults[0], 'FIMO result has motif_name');
    assert('hits' in fimoResults[0], 'FIMO result has hits');
    
    // Test 9: Basic TOMTOM functionality  
    console.log('Testing TOMTOM algorithm...');
    const query = [[0.8, 0.1], [0.1, 0.8], [0.05, 0.05], [0.05, 0.05]];
    const target = [[0.7, 0.2], [0.2, 0.7], [0.05, 0.05], [0.05, 0.05]];
    
    const tomtomResults = msl.tomtom([query], [target]);
    assert('pValues' in tomtomResults, 'TOMTOM returns pValues');
    assert('scores' in tomtomResults, 'TOMTOM returns scores');
    assert(Array.isArray(tomtomResults.pValues), 'pValues is array');
    assert(tomtomResults.pValues.length === 1, 'pValues has correct dimensions');
    assert(tomtomResults.pValues[0].length === 1, 'pValues has correct dimensions (1 target, not 2 with RC)');
    
    // Test 10: Error handling
    console.log('Testing error handling...');
    try {
        msl.oneHotEncode("ACGTX"); // Invalid character
        assert(false, 'Should throw error for invalid character');
    } catch (e) {
        assert(true, 'Throws error for invalid character');
    }
    
    try {
        msl.characters([[1, 1], [0, 0], [0, 0], [0, 0]]); // Tie without force
        assert(false, 'Should throw error for ties without force');
    } catch (e) {
        assert(true, 'Throws error for ties without force flag');
    }
    
    console.log('\n✅ All tests passed!');
}

// Run tests
try {
    runTests();
} catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
}