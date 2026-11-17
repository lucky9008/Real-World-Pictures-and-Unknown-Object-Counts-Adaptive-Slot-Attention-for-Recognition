# Quantum Adaptive Slot Attention (QASA)

Code accompanying the paper **"Real-World Pictures and Unknown Object Counts: Adaptive Slot Attention for Recognition"**
(Snigdh Rawal, Uday Pundir, Ravi Kumar Ardhana, Giandeep Singh; UPES, Dehradun, India).
The paper is included in [`paper/`](paper/Adaptive_Slot_Attention_QASA_final.pdf).

The paper compares global self-attention, fixed Slot Attention, AdaSlot, and QASA, a hybrid pipeline that uses
(simulated) Quantum Counting to estimate how many of `K_max = 16` candidate slots to keep. On frozen DINO ViT-S/16
features, adaptive methods keep 4-8 of 16 slots and retain 74%-99% of the fine-tuned 16-slot FG-ARI over three seeds.
Simulated quantum counting gave no consistent accuracy advantage over AdaSlot and needs more oracle calls than direct
counting at `K_max = 16`.

## Repository layout

```
.
├── models/
│   ├── classical.py                # SlotAttention, MultiHeadSelfAttention, SpatialBroadcastDecoder
│   └── quantum_slot_attention.py   # QuantumRefinementBlock, HybridQuantumSlotAttention (PennyLane VQC)
├── paper/
│   └── Adaptive_Slot_Attention_QASA_final.pdf
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/qasa-slot-attention.git
cd qasa-slot-attention
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## What is in this release

| Module | Description |
| --- | --- |
| `SlotAttention` | Competitive Slot Attention (softmax over slots, weighted mean, GRU, residual MLP). Slot count `K` and iterations `T` can be overridden per call; an optional `seed` makes slot initialisation reproducible. |
| `MultiHeadSelfAttention` | Standard multi-head self-attention over patch tokens (the global-attention baseline). |
| `SpatialBroadcastDecoder` | Broadcasts each slot over a 14x14 grid, upsamples to 224x224 and returns per-slot RGB, soft alpha masks (softmax over slots) and the reconstruction. |
| `QuantumRefinementBlock` | Variational quantum circuit (AngleEmbedding + StronglyEntanglingLayers, 4 qubits, 2 layers) used as a residual refinement step. Runs on PennyLane's `default.qubit` simulator. |
| `HybridQuantumSlotAttention` | Slot Attention where the residual MLP is replaced by `QuantumRefinementBlock` after each GRU update. Also returns the Pauli-Z expectation values and total quantum latency (ms). |

## Quick example

```python
import torch
from models import SlotAttention, SpatialBroadcastDecoder, HybridQuantumSlotAttention

feats = torch.randn(2, 196, 256)          # (batch, N patches, D)

sa = SlotAttention(num_slots=16, dim=256, iters=3, input_dim=256)
slots, attn, all_attns = sa(feats, seed=0)

hq = HybridQuantumSlotAttention(num_slots=4, dim=256, input_dim=256, differentiable=True)
slots, attn, all_attns, expvals, q_latency_ms = hq(feats[:1], seed=0)
```

The quantum block simulates one circuit per slot per iteration on the CPU, so it is slow for large batches or many slots.

## Important: scope of this code release vs. the paper

Please read this before trying to reproduce the paper's numbers.

- **Included:** the Slot Attention, baseline attention, decoder and the PennyLane quantum-refinement modules.
- **Not included yet:** the experiment pipeline described in Sections VI-VII of the paper, i.e. DINO feature extraction,
  the keep/drop selector, the simulated Quantum Counting routine (Eq. 21), the dual-ascent budget controller,
  training scripts, and the FG-ARI / keep-drop evaluation code. Tables III-IX and Figures 5-13 cannot be regenerated from
  this repository as it stands.
- **Two different quantum components.** The paper's QASA uses *simulated Quantum Counting* to choose the number of active
  slots. `HybridQuantumSlotAttention` in this repository is a different construction: a variational quantum circuit that
  refines slot vectors in place of the residual MLP. It is not the counting stage evaluated in the paper.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). A BibTeX entry will be added once the
paper has a venue/arXiv identifier.

## License

MIT, see [`LICENSE`](LICENSE).
