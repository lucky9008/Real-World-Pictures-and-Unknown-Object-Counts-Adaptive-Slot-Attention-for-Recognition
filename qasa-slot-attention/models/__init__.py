from .classical import MultiHeadSelfAttention, SlotAttention, SpatialBroadcastDecoder

__all__ = [
    "SlotAttention",
    "MultiHeadSelfAttention",
    "SpatialBroadcastDecoder",
    "QuantumRefinementBlock",
    "HybridQuantumSlotAttention",
]


def __getattr__(name):
    # PennyLane is only needed for the quantum modules, so import lazily.
    if name in ("QuantumRefinementBlock", "HybridQuantumSlotAttention"):
        from . import quantum_slot_attention

        return getattr(quantum_slot_attention, name)
    raise AttributeError(name)
