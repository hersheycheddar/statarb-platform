# Statistical Arbitrage Platform

## Project Goal
Modular statistical arbitrage & pairs trading backtesting engine.

## Core Engineering Rule: No Lookahead Bias
Strictly avoid lookahead bias. All rolling calculations, signals, and features
must use only past data. Any rolling/expanding computation whose result is
used to generate a trading decision must be `.shift(1)`-ed (or otherwise
lagged) before being applied to that same period's returns. When in doubt,
assume a value is not yet known until the *next* bar.

## Code Standards
- Python 3.10+ syntax and features.
- Strict type annotations on all function signatures and public class
  attributes (no bare `Any` unless truly dynamic).
- Every function/class has a docstring explaining the *statistical rationale*
  (why the method is used, not just what it does).
- Modular design: separate data loading, signal generation, backtesting
  execution, and performance analytics into distinct modules.
- Every module has corresponding pytest test coverage.

## Commands

Activate the environment:
```bash
source .venv/bin/activate
```

Install/update dependencies:
```bash
pip install -r requirements.txt
```

Run the full test suite:
```bash
pytest
```

Run a single test file:
```bash
pytest tests/test_<module>.py
```

Run a script/module:
```bash
python -m src.<module_name>
```
