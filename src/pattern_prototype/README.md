# Pattern-Constrained Model Prototype

This is a prototype implementation of a pattern-constrained language model for the Two-Button Keyboard project.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
# Test keyboard mapping
python keyboard_mapping.py

# Test full model
python test_model.py
```

## Components

- `keyboard_mapping.py`: Maps characters to L/R patterns based on QWERTY keyboard layout
- `model.py`: Pattern-constrained transformer model implementation
- `test_model.py`: Integration tests and examples

## Testing

The test script will log results to the console. You should see output showing:
1. Pattern matching tests
2. Model architecture tests
3. Keyboard integration tests

If any test fails, detailed error information will be logged.