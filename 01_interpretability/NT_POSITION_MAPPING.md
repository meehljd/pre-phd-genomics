# Nucleotide Transformer Position Mapping

## Issue

When analyzing variants with Nucleotide Transformer, there's a position mismatch:

- **Variant position**: Amino acid position (e.g., AA 248)
- **Attention matrix**: Token positions (e.g., 198 tokens)

## Why This Happens

1. **AA to NT conversion**: AA position 248 → NT position 744 (248 × 3)
2. **6-mer tokenization**: Nucleotide Transformer uses 6-mer tokens
   - Input: "ATCGATCG..." (744 nt)
   - Tokens: ~124 tokens (744 ÷ 6)
3. **Special tokens**: [CLS], [SEP] tokens may be added
4. **Result**: Token count doesn't match AA position

## Current Solution

The `analyze_variant_attention_changes()` function now includes bounds checking:
- If `variant_pos` exceeds attention matrix size, it uses sequence center
- Prints a warning explaining the mismatch

## Better Approaches

### Option 1: Global Analysis (Recommended)
Don't specify a variant position - analyze the entire sequence:

```python
changes = analyze_variant_attention_changes(
    wt_attn, 
    mut_attn, 
    variant_pos=None,  # Analyze whole sequence
    variant_type=variant_type
)
```

### Option 2: Calculate Token Position
Map AA position to approximate token position:

```python
def aa_pos_to_token_pos(aa_pos, seq_length_nt, num_tokens):
    """
    Approximate token position from amino acid position.
    
    Args:
        aa_pos: 1-indexed amino acid position
        seq_length_nt: Total nucleotide sequence length
        num_tokens: Number of tokens in attention matrix
    
    Returns:
        Approximate token index (0-indexed)
    """
    nt_pos = (aa_pos - 1) * 3  # Convert to 0-indexed NT position
    token_pos = int((nt_pos / seq_length_nt) * num_tokens)
    return min(token_pos, num_tokens - 1)
```

### Option 3: Use Protein Models
For amino-acid level attention analysis, use protein language models:
- ESM-2
- ProtTrans
- ESM-1b

These work directly with AA sequences and positions.

## Example

```python
# TP53 R248W
variant = PATHOGENIC_VARIANTS[3]
print(f"AA position: {variant['pos']}")  # 248
print(f"NT position: {variant['pos'] * 3}")  # 744
print(f"Token position: ~{744 // 6}")  # ~124

# But attention matrix might be different size!
wt_attn = extract_attention(model, tokenizer, wt_seq, ...)
print(f"Attention matrix size: {wt_attn.shape[0]}")  # e.g., 198

# Solution: Use center or calculate mapping
token_pos = aa_pos_to_token_pos(248, len(wt_seq), wt_attn.shape[0])
```

## Recommendation

For Nucleotide Transformer analysis:
1. Focus on **global** sequence changes, not position-specific
2. Compare overall attention patterns between WT and mutant
3. Use layer-wise analysis to see which layers change most
4. For position-specific analysis, switch to protein models (ESM-2)
