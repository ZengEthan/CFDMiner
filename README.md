# CFD Miner (CCFDMiner)

A lightweight Python implementation for mining **Conditional Functional Dependencies (CFDs)** from categorical data in CSV format.

This project parses a CSV dataset into transactions, mines frequent itemsets using **bitset (bitarray) TID-lists** for fast intersections, derives **minimal generators and closures**, and prints CFD-like rules to stdout via Python logging.

> Source file: `CFDMiner.py`. fileciteturn0file0

---

## Features

- **CSV → transactional encoding** (attribute=value tokens mapped to integers)
- **Bitset-backed TID-lists** using `bitarray` for fast AND intersections
- Mines **frequent itemsets** (by minimum support)
- Computes **minimal generators** and their **closures**
- Prints discovered rules using the internal token mapping (attribute and value names)

---

## Installation

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

The script is designed to be run from the command line:

```bash
python CFDMiner.py <infile.csv> <minsupp> <maxsize>
```

Where:

- `infile.csv` is a CSV dataset
- `minsupp` is a positive integer minimum support threshold (number of rows/transactions)
- `maxsize` is a positive integer maximum size of the left-hand side (number of attributes/items in a rule)

Example:

```bash
python CFDMiner.py data.csv 10 4
```

### Input format

- The CSV is assumed to be **categorical**.
- By default, the first row is treated as **headers** (attribute names).
- Each cell becomes a token like `Attr=Value`.

If you need to change delimiter or header behavior, edit the `load_csv()` call in `main()`.

---

## Output

Rules are printed via the Python `logging` module (INFO level). A typical rule line looks like:

```
[AttrA, AttrB] => AttrC,(valA ,valB || valC)
```

Interpretation (informal):

- If a row matches the LHS attribute-values (the generator),
- then the RHS attribute-value(s) in the closure are implied for rows in the same equivalence class (support group).

At the end, the script logs:

- number of generators found
- total number of rules printed
- a global intersection counter (useful for performance profiling)

---

## How it works (high level)

1. **Tokenization**
   - Each `(attribute, value)` pair is mapped to a unique integer item ID.
2. **TID-lists as bitsets**
   - For every item, a bitarray marks which transactions contain the item.
3. **Depth-first mining**
   - Recursively intersects bitsets to create larger itemsets, pruning by `minsupp`.
4. **Minimal generators & closure**
   - Tracks generators in a hash-indexed map and updates closures based on joins.

---

## Requirements

- Python **3.9+** recommended (uses modern type hints like `list[int]`)
- Dependencies are listed in `requirements.txt`.

---

## Notes & limitations

- Designed for **categorical** data; numeric columns should be discretized first.
- Empty cells are treated as a literal empty string value (after `.strip()`).
- Output formatting is logging-based; redirect stdout/stderr if you want to save results:
  ```bash
  python CFDMiner.py data.csv 10 4 > rules.log 2>&1
  ```

---

## License

Add a license if you plan to distribute this code publicly.
