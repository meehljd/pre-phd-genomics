# DNABERT-2 CUDA Setup Summary

## Problem
DNABERT-2 requires Flash Attention (Triton), which has compatibility issues:
1. **Triton version conflicts**: DNABERT-2's Flash Attention code uses deprecated `trans_b` parameter
2. **GPU compatibility**: Flash Attention requires specific GPU architectures (GTX 970 is too old)
3. **CPU incompatibility**: Flash Attention requires CUDA, doesn't work on CPU

## Solution Applied

### 1. Disabled Flash Attention
Patched `/root/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT_hyphen_2_hyphen_117M/*/bert_layers.py`:

```python
# CPU-COMPATIBLE PATCH: Disable Flash Attention
try:
    # from .flash_attn_triton import flash_attn_qkvpacked_func
    flash_attn_qkvpacked_func = None  # Force standard PyTorch attention
except:
    flash_attn_qkvpacked_func = None
```

### 2. Uninstalled Triton
```bash
pip uninstall triton -y
```

### 3. Updated Notebook Device Selection
Modified [02_dnabert2_attention_analysis.ipynb](02_dnabert2_attention_analysis.ipynb):
- Cell 2: Updated device selection to prefer CUDA > MPS > CPU
- Cell 3: Removed `output_attentions=True` from `from_pretrained()` (not supported by DNABERT-2)

## Current Status

✓ **DNABERT-2 runs on CUDA** (GTX 970)
✓ **Model loads successfully**
✓ **Forward pass works**
✗ **Attention extraction NOT working with standard PyTorch backend**

###Output Format
```python
outputs = model(**inputs, output_attentions=True)
# Returns: (last_hidden_state, pooler_output)
# Shape: (torch.Size([1, seq_len, 768]), torch.Size([1, 768]))
```

**Issue**: Standard PyTorch attention backend in DNABERT-2 doesn't return attention matrices.

## Next Steps (Options)

### Option 1: Use a Different DNA Model
- **NT-Transformer** or **Nucleotide Transformer** - designed for attention analysis
- **HyenaDNA** - may have better attention extraction support

### Option 2: Fix DNABERT-2 Attention Extraction
Modify the standard PyTorch attention code in `bert_layers.py` to return attention matrices

### Option 3: Use Full Sequences Without Attention
- Extract embeddings from DNABERT-2
- Use gradient-based attribution methods instead of attention

### Option 4: Upgrade GPU/Environment
- Use a machine with newer GPU (Ampere/Hopper architecture)
- Flash Attention 2 works better on modern GPUs

## Recommended: Option 1

Switch to **Nucleotide Transformer** which is:
- Designed for genomic sequences
- Has better documentation
- Standard Transformers architecture with proper attention output

## Testing

To test current setup:
```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, use_safetensors=True)

device = torch.device("cuda")
model = model.to(device)
model.eval()

inputs = tokenizer("ATCGATCG", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(f"Output shape: {outputs[0].shape}")  # Works!
```

## Warning Message (Expected)
```
UserWarning: Unable to import Triton; defaulting MosaicBERT attention implementation to pytorch
```

This is normal and expected after disabling Flash Attention.
