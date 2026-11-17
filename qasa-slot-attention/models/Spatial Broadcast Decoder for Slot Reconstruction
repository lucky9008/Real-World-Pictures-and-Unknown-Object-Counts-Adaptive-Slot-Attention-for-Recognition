class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int = 64, hidden_dim: int = 64, initial_size: Tuple[int, int] = (14, 14)):
        super().__init__()
        self.initial_size = initial_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 4, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1),
        )

    def forward(self, slots: torch.Tensor, target_size: Tuple[int, int] = (224, 224)):
        b, k, d = slots.shape
        h0, w0 = self.initial_size

        # 1. Broadcast slots over spatial grid
        slots_flat = slots.reshape(b * k, d, 1, 1).expand(-1, -1, h0, w0)

        # 2. Append linear coordinate grids [-1, 1]
        y = torch.linspace(-1.0, 1.0, h0, device=slots.device)
        x = torch.linspace(-1.0, 1.0, w0, device=slots.device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gx, gy, 1.0 - gx, 1.0 - gy], dim=0).unsqueeze(0).expand(b * k, -1, -1, -1)

        dec_in = torch.cat([slots_flat, coords], dim=1)
        dec_out = self.net(dec_in)
        dec_out = dec_out.reshape(b, k, 4, target_size[0], target_size[1])

        slot_rgb = dec_out[:, :, :3]
        mask_logits = dec_out[:, :, 3:]
        alpha = F.softmax(mask_logits, dim=1)  # Softmax competition across slots
        recon = (slot_rgb * alpha).sum(dim=1)

        return recon, slot_rgb, alpha
