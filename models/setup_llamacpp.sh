#!/bin/bash

# Setup script for llama.cpp with Metal acceleration

# Clone llama.cpp
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
else
    cd llama.cpp
    git pull
fi

# Build with Metal support
LLAMA_METAL=1 make

# Create phi2 directory if it doesn't exist
mkdir -p ../phi2

# Download and convert Phi-2 model
echo "Next steps:"
echo "1. Download Phi-2 model from HuggingFace:"
echo "   https://huggingface.co/microsoft/phi-2"
echo "2. Convert model:"
echo "   python3 convert.py --outfile ../phi2/phi2-q4_k.gguf \
         --outtype q4_k /path/to/downloaded/phi2"
echo "3. Test model:"
echo "   ./main -m ../phi2/phi2-q4_k.gguf \
         -n 256 --metal --prompt 'Write a haiku about:'"