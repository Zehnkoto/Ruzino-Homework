"""
pytest configuration for TreeGen tests
Sets up paths and environment variables
"""
import sys
import os

# Get binary directory
# From source/Plugins/TreeGen/tests/ to Binaries/Debug
tests_dir = os.path.dirname(os.path.abspath(__file__))
binary_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Binaries", "Release"))

# Set PXR_USD_WINDOWS_DLL_PATH so USD can find its DLLs
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = binary_dir
print(f"Set PXR_USD_WINDOWS_DLL_PATH={binary_dir}")

# Add to Python path
sys.path.insert(0, binary_dir)

# Add rznode python path
# From source/Plugins/TreeGen/tests/ to source/Core/rznode/python
rznode_python = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "Core", "rznode", "python"))
sys.path.insert(0, rznode_python)

# Change to binary dir so DLLs can be loaded
os.chdir(binary_dir)
print(f"Changed working directory to: {os.getcwd()}")
