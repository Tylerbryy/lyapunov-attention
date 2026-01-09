# Lyapunov Exponents for Attention Composition

Experimental verification code for the paper:

**"Lyapunov Exponents for Attention Composition: A Dynamical Systems Perspective on Deep Transformers"**

Tyler Gibbs, Backwork AI

## Overview

This repository contains the code to reproduce all experimental results in the paper. We develop the first Lyapunov exponent framework for analyzing eigenvalue dynamics in composed attention layers, bridging transformer theory with dynamical systems.

### Key Contributions

1. **First Lyapunov spectrum computation for attention** - Proving Lambda_1 = 0 exactly and Lambda_k < 0 for k > 1
2. **Temperature-spectral gap relationship** - Quantifying how softmax temperature affects collapse rates
3. **Refined collapse prediction formula** - L_collapse = log((r-1)/(d-1)) / log(gamma)
4. **Non-commutative Lyapunov insight** - Attention Lyapunov exponents differ from naive eigenvalue products
5. **Residual connection mechanism** - Residual connections reduce |Lambda_2| by ~2.4x

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
python lyapunov_attention.py
```

This will reproduce all tables from the paper:
- Table 1: Lyapunov spectrum (d=50, T=1.0, L=100)
- Table 2: Non-commutative structure comparison
- Table 3: Temperature effect on spectral collapse
- Table 4: Rank collapse prediction validation
- Table 5: Residual connection effect

### Using as a Library

```python
from lyapunov_attention import (
    generate_attention_matrix,
    compute_lyapunov_spectrum,
    effective_rank
)

# Generate random attention matrix
A = generate_attention_matrix(n=50, temperature=1.0)

# Compute Lyapunov spectrum for 100-layer composition
results = compute_lyapunov_spectrum(n=50, n_layers=100, temperature=1.0)
print(f"Lambda_1 = {results['mean'][0]:.6f}")  # Should be ~0
print(f"Lambda_2 = {results['mean'][1]:.6f}")  # Should be negative
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gibbs2026lyapunov,
  title={Lyapunov Exponents for Attention Composition: A Dynamical Systems Perspective on Deep Transformers},
  author={Gibbs, Tyler},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

Tyler Gibbs - tylergibbs@backworkai.com
