# SAR System - Windows PowerShell Setup Guide

## Overview

This guide provides correct Python execution commands for Windows PowerShell, eliminating module-loading errors and path issues.

## Virtual Environment

The project uses a virtual environment located at `.venv` in the project root.

### Quick Start (Recommended)

```powershell
# Navigate to project root
cd c:\Users\Manendra\Desktop\innovatron

# Activate the virtual environment
.\activate-venv.ps1

# You should see (venv) in your prompt

# Install dependencies (only needed once)
pip install -r requirements.txt

# Navigate to decision_making
cd aerial_disaster_mngmt\decision_making

# Run the application
python main.py

# Exit venv when done
deactivate
```

### Without Activation (Direct Execution)

If you prefer not to activate the venv, use the absolute path to the Python executable:

```powershell
# From project root
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run verification
.venv\Scripts\python.exe verify_setup.py

# Run application
cd aerial_disaster_mngmt\decision_making
.venv\Scripts\python.exe main.py
```

## Correct Commands

### ✓ DO USE (Correct)

```powershell
# With activated venv
python main.py
python -m pip install numpy

# With absolute path and call operator (&)
& ".venv\Scripts\python.exe" main.py
.venv\Scripts\python.exe -c "import sys; print(sys.executable)"

# In PowerShell scripts
& $pythonExecutable "script.py"
```

### ✗ DO NOT USE (Incorrect)

```powershell
# Relative paths with multiple ..
.venv\..\..\venv\Scripts\python.exe

# Bare relative paths that PowerShell tries to import as modules
python.exe main.py  # Use "python" not "python.exe"
.venv\Scripts\python.exe main.py  # Use & ".venv\Scripts\python.exe" instead

# Mixed path separators
.venv/Scripts/python.exe  # Use backslashes only on Windows
```

## File Structure

```
innovatron/
├── .venv/                              # Virtual environment (do not commit)
│   ├── Scripts/
│   │   ├── Activate.ps1               # Activation script for PowerShell
│   │   ├── python.exe                 # Python executable
│   │   └── pip.exe                    # Package manager
│   └── Lib/                            # Installed packages
├── .vscode/
│   └── settings.json                  # VS Code configuration (uses .venv)
├── activate-venv.ps1                  # Helper to activate venv
├── verify_setup.py                    # Verification script
├── requirements.txt                   # Python dependencies
└── aerial_disaster_mngmt/
    ├── interfaces/
    │   └── perception_schema.py
    ├── perception/
    │   ├── perception_adapters.py
    │   ├── perception_utils.py
    │   └── ...
    └── decision_making/
        ├── main.py
        ├── fusion_engine.py
        └── ...
```

## Verification

### Verify Python Executable

```powershell
cd c:\Users\Manendra\Desktop\innovatron

# After activation
.\activate-venv.ps1
python -c "import sys; print(sys.executable)"
# Output should show: C:\Users\Manendra\Desktop\innovatron\.venv\Scripts\python.exe

deactivate
```

### Run Comprehensive Verification

```powershell
cd c:\Users\Manendra\Desktop\innovatron

# Option 1: With activation
.\activate-venv.ps1
python verify_setup.py

# Option 2: Without activation
.venv\Scripts\python.exe verify_setup.py
```

### Expected Output

```
======================================================================
  SAR SYSTEM - ENVIRONMENT SETUP VERIFICATION
======================================================================

[OK] Virtual environment directory (.venv)
     -> C:\Users\Manendra\Desktop\innovatron\.venv
[OK] Python executable (python.exe)
     -> C:\Users\Manendra\Desktop\innovatron\.venv\Scripts\python.exe
[OK] Python version
     -> 3.13.5
...
[OK] Environment
[OK] Python Executable
[OK] Installed Packages
[OK] Project Structure
[OK] Key Files
[OK] Module Imports

======================================================================
  6/6 checks passed
======================================================================

[OK] Environment is properly configured!
```

## Common Issues

### Issue: "The module '.venv' could not be loaded"

**Cause**: PowerShell tried to load `.venv` as a module instead of executing the path.

**Solution**: Use the call operator (`&`) or activate the venv first.

```powershell
# Wrong
.venv\Scripts\python.exe main.py

# Correct
& ".venv\Scripts\python.exe" main.py
# OR
.\activate-venv.ps1
python main.py
```

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Cause**: Running Python from outside the venv without using the absolute venv path.

**Solution**: Activate the venv or use the absolute path.

```powershell
# Activate first
.\activate-venv.ps1
python main.py

# Or use absolute path
.venv\Scripts\python.exe main.py
```

### Issue: "No module named 'logger_config'"

**Cause**: Running from wrong directory or with incorrect sys.path manipulation.

**Solution**: Run from the correct directory.

```powershell
# Correct
cd aerial_disaster_mngmt\decision_making
python main.py

# Or with absolute path
.venv\Scripts\python.exe -C aerial_disaster_mngmt/decision_making main.py
```

## Environment Variables

The venv automatically sets:

- `PATH` - includes `.venv\Scripts`
- `VIRTUAL_ENV` - points to `.venv`
- `PYTHONHOME` - unset to use venv's Python

These are only active when the venv is **activated**. Use absolute paths if not activating.

## VS Code Integration

VS Code is configured to use `.venv\Scripts\python.exe` as the Python interpreter via `.vscode/settings.json`.

- Open the command palette: `Ctrl+Shift+P`
- Type "Python: Select Interpreter"
- Choose the one pointing to `.venv\Scripts\python.exe`

The interpreter is automatically detected from settings.json.

## Troubleshooting

If you still see PowerShell errors:

1. **Delete and recreate the venv**:
   ```powershell
   Remove-Item -Recurse -Force .venv
   python -m venv .venv
   pip install -r requirements.txt
   ```

2. **Check PowerShell execution policy**:
   ```powershell
   Get-ExecutionPolicy
   # If 'Restricted', allow scripts:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Use absolute paths consistently**:
   ```powershell
   $pythonPath = (Resolve-Path ".venv\Scripts\python.exe").Path
   & $pythonPath main.py
   ```

## Summary

| Task | Command |
|------|---------|
| Activate venv | `.\activate-venv.ps1` |
| Deactivate venv | `deactivate` |
| Install packages | `pip install -r requirements.txt` (after activation) |
| Run tests | `python verify_setup.py` (after activation) |
| Run main app | `cd aerial_disaster_mngmt\decision_making` then `python main.py` |
| Run without activation | `.venv\Scripts\python.exe main.py` |

---

**Last Updated**: January 29, 2026
**Python Version**: 3.13.5
**Virtual Environment**: .venv
