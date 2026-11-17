class SlotAttention(nn.Module):
    """Competitive Slot Attention routing module."""

    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        in_d = input_dim or dim

        # Gaussian slot distribution parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(in_d, dim, bias=False)
        self.to_v = nn.Linear(in_d, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(in_d)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(
        self,
        inputs: torch.Tensor,
        num_slots: Optional[int] = None,
        iters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        b, n, _ = inputs.shape
        k = num_slots or self.num_slots
        t_iters = iters or self.iters

        # Adaptable slot count (K) sampling
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

        for _ in range(t_iters):
            slots_norm = self.norm_slots(slots)
            q_proj = self.to_q(slots_norm)

            # Competitive Softmax across slots (dim=1)
            dots = torch.einsum("bkd,bnd->bkn", q_proj, k_proj) * self.scale
            attn = F.softmax(dots, dim=1)
            all_attns.append(attn)

            # Weighted mean
            attn_norm = attn + self.eps
            attn_norm = attn_norm / attn_norm.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v_proj)

            # GRU update + Residual MLP refinement
            slots = self.gru(updates.reshape(-1, self.dim), slots.reshape(-1, self.dim))
            slots = slots.reshape(b, k, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn, all_attns
