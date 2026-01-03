# Windows Build Tools Installation Guide

If you encounter build errors when installing Python packages, you may need to install Microsoft Visual C++ Build Tools.

## Quick Fix (Recommended)

### Option 1: Use Pre-compiled Wheels
```bash
# Install packages that don't require compilation
pip install -r requirements-windows.txt
```

### Option 2: Install Build Tools
1. Download **Microsoft C++ Build Tools**: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select **"C++ build tools"** workload
4. Install and restart your computer
5. Run the setup script again

### Option 3: Use Conda (Alternative)
```bash
# Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
conda create -n frigate python=3.11
conda activate frigate
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements-windows.txt
```

## Common Build Errors

### pysqlite3 Error
```
error: Microsoft Visual C++ 14.0 or greater is required
```
**Solution**: Use built-in SQLite (already fixed in requirements-windows.txt)

### cython-bbox Error
```
error: command 'cl.exe' failed with exit code 2
```
**Solution**: Install Visual C++ Build Tools or use pre-compiled wheels

### Other C Extension Errors
**Solution**: Install Visual C++ Build Tools or use conda

## Package Alternatives

If you can't install build tools, use these alternatives:

```bash
# Instead of pysqlite3, use built-in sqlite3
# (Already handled in requirements-windows.txt)

# Instead of cython-bbox, install separately if needed
pip install cython-bbox --only-binary=all

# For other problematic packages, try:
pip install package-name --only-binary=all
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
venv\Scripts\activate.bat

# Test imports
python -c "import frigate; print('Frigate OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import pyclipper; print('PyClipper OK')"
```

## Still Having Issues?

1. **Use the minimal requirements**: `pip install -r requirements-minimal.txt`
2. **Install packages one by one**: Identify which specific package is failing
3. **Use conda**: Often handles build dependencies automatically
4. **Check package documentation**: Some packages have Windows-specific installation instructions
