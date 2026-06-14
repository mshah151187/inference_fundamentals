#!/bin/bash
# Run this at the start of every new Lambda Labs session.
# Installs everything not pre-installed by Lambda Stack 22.04.

set -e  # stop on first error

echo "=== Installing Python packages ==="

# core ML
pip install --upgrade Pillow
pip install transformers accelerate

# ZMQ — inter-process communication between pipeline stages
pip install pyzmq

# protobuf + grpc-tools — schema-based serialization + code generation
# grpcio-tools includes the protoc compiler accessible via:
#   python -m grpc_tools.protoc -I proto --python_out=generated proto/messages.proto
pip install grpcio grpcio-tools protobuf

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

echo ""
echo "=== Instance ready ==="
