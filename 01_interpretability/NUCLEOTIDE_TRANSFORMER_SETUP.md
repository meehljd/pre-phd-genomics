# Nucleotide Transformer Setup - SUCCESS!

## Overview

Successfully set up **Nucleotide Transformer** for genomic variant attention analysis. This model has a standard transformer architecture with proper attention extraction support.

## Model Details

**Model**: `InstaDeepAI/nucleotide-transformer-500m-human-ref`

### Architecture
- **Parameters**: 500M
- **Layers**: 24
- **Attention Heads**: 20 per layer
- **Hidden Size**: 1280
- **Training Data**: Human reference genome
- **Tokenization**: 6-mers (vocabulary size: 4105)

### Advantages over DNABERT-2
✓ Standard transformer architecture (no Flash Attention issues)
✓ Proper attention extraction works out of the box
✓ Returns attention matrices for all layers
✓ Compatible with all GPUs (tested on GTX 970)
✓ Well-documented and maintained
✓ Published in Nature Methods 2025

## Installation

```bash
# Already installed with transformers
pip install transformers[torch]
```

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model
model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(
    model_name,
    output_attentions=True,
    use_safetensors=True,  # Required for PyTorch 2.5+
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Run inference
sequence = "ATCGATCGATCGATCGATCG"
inputs = tokenizer(sequence, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Access attentions
attentions = outputs.attentions  # Tuple of 24 tensors
# Each: [batch=1, heads=20, seq_len, seq_len]

print(f"Number of layers: {len(attentions)}")
print(f"First layer shape: {attentions[0].shape}")
```

## Output Format

```python
outputs = model(**inputs, output_attentions=True)

# Returns MaskedLMOutput with:
outputs.logits          # [batch, seq_len, vocab_size]
outputs.hidden_states   # If output_hidden_states=True
outputs.attentions      # Tuple of 24 attention tensors
                        # Each: [batch, heads, seq_len, seq_len]
```

## Integration with Variant Dataset

The nucleotide sequences from [variant_dataset.py](variant_dataset.py) can be used directly:

```python
from variant_dataset import PATHOGENIC_VARIANTS, get_variant_nt_sequence

# Get wild-type and mutant sequences
variant = PATHOGENIC_VARIANTS[0]  # HBB E7V
wt_seq = get_variant_nt_sequence(variant, version='wt')
mut_seq = get_variant_nt_sequence(variant, version='mut')

# Tokenize and run through model
wt_inputs = tokenizer(wt_seq, return_tensors="pt").to(device)
mut_inputs = tokenizer(mut_seq, return_tensors="pt").to(device)

with torch.no_grad():
    wt_outputs = model(**wt_inputs, output_attentions=True)
    mut_outputs = model(**mut_inputs, output_attentions=True)

# Compare attentions
wt_attn = wt_outputs.attentions[-1]  # Last layer
mut_attn = mut_outputs.attentions[-1]
diff = (mut_attn - wt_attn).abs().mean()
print(f"Mean attention difference: {diff:.6f}")
```

## Key Differences from DNABERT-2

| Feature | DNABERT-2 | Nucleotide Transformer |
|---------|-----------|------------------------|
| Flash Attention | Required | Not used |
| Triton dependency | Yes | No |
| Attention extraction | Broken with standard backend | Works perfectly |
| GPU compatibility | Modern GPUs only | All CUDA GPUs |
| CPU support | Broken | Works |
| Output format | Tuple (hidden, pooler) | Standard MaskedLMOutput |
| Tokenization | K-mers | 6-mers |

## Next Steps

1. ✓ Model loads and runs on CUDA
2. ✓ Attention extraction verified
3. **TODO**: Create Jupyter notebook for variant attention analysis
4. **TODO**: Test with actual variant sequences
5. **TODO**: Visualize attention patterns for pathogenic vs benign variants
6. **TODO**: Compare attention changes across different layers

## References

- **Paper**: Dalla-Torre et al., "Nucleotide Transformer: building and evaluating robust foundation models for human genomics", Nature Methods (2025)
- **GitHub**: https://github.com/instadeepai/nucleotide-transformer
- **HuggingFace**: https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref
- **Documentation**: https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb
