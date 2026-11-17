import math
import time
from typing import Any, Dict, List, Optional, Tuple
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumRefinementBlock(nn.Module):
    """Variational Quantum Circuit replacing classical residual MLP refinement."""

    def __init__(
        self,
        slot_dim: int = 64,
        num_qubits: int = 4,
        n_layers: int = 2,
        differentiable: bool = False,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.differentiable = differentiable

        # Classical projections
        self.norm = nn.LayerNorm(slot_dim)
        self.linear_down = nn.Linear(slot_dim, num_qubits)
        self.linear_up = nn.Linear(num_qubits, slot_dim)

        # Variational parameters: shape (n_layers, num_qubits, 3) = 24 params
        weight_shape = (n_layers, num_qubits, 3)
        self.q_weights = nn.Parameter(torch.empty(weight_shape))
        nn.init.uniform_(self.q_weights, -math.pi, math.pi)

        # PennyLane statevector simulator
        self.dev = qml.device("default.qubit", wires=num_qubits)
        diff_method = "backprop" if differentiable else None

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit(inputs: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self._circuit = _circuit

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        b, k, d = slots.shape
        orig_device = slots.device
        orig_dtype = slots.dtype

        # Down-projection & bounded angular encoding [-pi, pi]
        slots_norm = self.norm(slots)
        angles = self.linear_down(slots_norm)
        angles = torch.tanh(angles) * math.pi
        flat_angles = angles.reshape(-1, self.num_qubits)

        t_start = time.perf_counter()

        if self.differentiable and torch.is_grad_enabled():
            flat_angles_cpu = flat_angles.to(device="cpu", dtype=torch.float64)
            weights_cpu = self.q_weights.to(device="cpu", dtype=torch.float64)
            res = torch.stack(self._circuit(flat_angles_cpu, weights_cpu), dim=-1)
            quantum_latency_ms = (time.perf_counter() - t_start) * 1000.0
            q_out = res.to(dtype=orig_dtype, device=orig_device).reshape(b, k, self.num_qubits)
        else:
            with torch.no_grad():
                flat_angles_cpu = flat_angles.detach().to(device="cpu", dtype=torch.float64)
                weights_cpu = self.q_weights.detach().to(device="cpu", dtype=torch.float64)
                res = torch.stack(self._circuit(flat_angles_cpu, weights_cpu), dim=-1)
                quantum_latency_ms = (time.perf_counter() - t_start) * 1000.0
                q_out = res.to(dtype=orig_dtype, device=orig_device).reshape(b, k, self.num_qubits)

        # Residual update
        refined_slots = slots + self.linear_up(q_out)
        return refined_slots, q_out, quantum_latency_ms


class HybridQuantumSlotAttention(nn.Module):
    """Slot Attention where GRU update is followed by Quantum Refinement."""

    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        input_dim: Optional[int] = None,
        num_qubits: int = 4,
        n_layers: int = 2,
        differentiable: bool = False,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        in_d = input_dim or dim

        # Gaussian slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(in_d, dim, bias=False)
        self.to_v = nn.Linear(in_d, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.norm_input = nn.LayerNorm(in_d)
        self.norm_slots = nn.LayerNorm(dim)

        self.quantum_refinement = QuantumRefinementBlock(
            slot_dim=dim,
            num_qubits=num_qubits,
            n_layers=n_layers,
            differentiable=differentiable,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: Optional[int] = None,
        iters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, float]:
        b, n, _ = inputs.shape
        k = num_slots or self.num_slots
        t_iters = iters or self.iters

        # Sample initial slots
        mu = self.slots_mu.expand(b, k, -1)
        sigma = self.slots_log_sigma.exp().expand(b, k, -1)
        if seed is not None:
            gen = torch.Generator(device=inputs.device).manual_seed(seed)
            slots = mu + sigma * torch.randn(mu.shape, generator=gen, device=inputs.device)
        else:
            slots = mu + sigma * torch.randn_like(mu)

        inputs_norm = self.norm_input(inputs)
        k_proj = self.to_k(inputs_norm)
        v_proj = self.to_v(inputs_norm)

        all_attns: List[torch.Tensor] = []
        attn = torch.zeros(b, k, n, device=inputs.device)
        total_q_lat = 0.0
        expvals = torch.zeros(b, k, self.quantum_refinement.num_qubits, device=inputs.device)

        for _ in range(t_iters):
            slots_norm = self.norm_slots(slots)
            q_proj = self.to_q(slots_norm)

            # Competitive Softmax over slots (dim=1)
            dots = torch.einsum("bkd,bnd->bkn", q_proj, k_proj) * self.scale
            attn = F.softmax(dots, dim=1)
            all_attns.append(attn)

            # Weighted mean
            attn_norm = attn + self.eps
            attn_norm = attn_norm / attn_norm.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v_proj)

            # GRU update
            slots = self.gru(updates.reshape(-1, self.dim), slots.reshape(-1, self.dim))
            slots = slots.reshape(b, k, self.dim)

            # Quantum Refinement
            slots, expvals, q_lat = self.quantum_refinement(slots)
            total_q_lat += q_lat

        return slots, attn, all_attns, expvals, total_q_lat