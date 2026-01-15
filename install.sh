#!/bin/bash
set -e

# Configuration
TORCH_VERSION="2.5.1"
CUDA_VERSION="cu121"
PYG_VERSION="2.7.0"  

echo "📋 Configuration:"
echo "   PyTorch version: ${TORCH_VERSION}"
echo "   CUDA version: ${CUDA_VERSION}"
echo "   PyG version: ${PYG_VERSION}"
echo ""

# Create/update conda environment
echo "🛠️  Setting up conda environment..."
conda env update -f environment.yml

# Activate environment
echo "🌟 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate gfm-discovery

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install --no-cache-dir \
  torch==${TORCH_VERSION} \
  --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install PyG extensions
echo "🌐 Installing PyTorch Geometric extensions..."
pip install --no-cache-dir \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install PyG
echo "🌐 Installing PyTorch Geometric..."
pip install torch-geometric==${PYG_VERSION}

# Verify
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo "✅ Setup complete!"