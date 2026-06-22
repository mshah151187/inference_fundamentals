#!/bin/bash
# Run this at the start of every new Lambda Labs session.
# Installs everything not pre-installed by Lambda Stack 22.04.
#
# Usage:
#   bash setup_instance.sh            # GPT-2 only (default)
#   bash setup_instance.sh --llama    # also download Llama 3.1 8B (requires HF_TOKEN)

set -e  # stop on first error

DOWNLOAD_LLAMA=0
for arg in "$@"; do
    if [ "$arg" == "--llama" ]; then
        DOWNLOAD_LLAMA=1
    fi
done

echo "=== Installing Python packages ==="

# core ML
pip install --upgrade Pillow
pip install "transformers>=4.43.0" accelerate

# bitsandbytes — INT8 / INT4-NF4 weight quantization for both GPT-2 and Llama
pip install bitsandbytes

# ZMQ — inter-process communication between pipeline stages
pip install pyzmq

# protobuf + grpc-tools — schema-based serialization + code generation
# grpcio-tools includes the protoc compiler accessible via:
#   python -m grpc_tools.protoc -I proto --python_out=generated proto/messages.proto
pip install grpcio grpcio-tools protobuf

# huggingface_hub CLI — needed for gated model downloads (Llama)
pip install huggingface_hub

echo ""
echo "=== Regenerating protobuf stubs ==="
# Regenerate generated/messages_pb2.py from proto/messages.proto.
# Run this whenever proto/messages.proto is changed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHING_DIR="$SCRIPT_DIR/../batching"
python -m grpc_tools.protoc \
    -I "$BATCHING_DIR/proto" \
    --python_out="$BATCHING_DIR/generated" \
    "$BATCHING_DIR/proto/messages.proto"
echo "Generated: batching/generated/messages_pb2.py"

echo ""
echo "=== Downloading GPT-2 model (if not already cached) ==="
python3 ~/inference_fundamentals/scripts/download_gpt2_model.py

if [ "$DOWNLOAD_LLAMA" == "1" ]; then
    echo ""
    echo "=== Downloading Llama 3.1 8B (requires HF_TOKEN) ==="
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN env var not set. Run: export HF_TOKEN=<your_token>"
        exit 1
    fi
    python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.pt", "original/*"],   # skip redundant pytorch bins
)
print("Llama 3.1 8B download complete.")
EOF
fi

echo ""
echo "=== Instance ready ==="
if [ "$DOWNLOAD_LLAMA" == "1" ]; then
    echo "    Workers available: gpt2, llama"
    echo "    Run: python main.py --worker llama"
else
    echo "    Worker available: gpt2 (default)"
    echo "    For Llama: re-run with HF_TOKEN set and --llama flag"
fi
