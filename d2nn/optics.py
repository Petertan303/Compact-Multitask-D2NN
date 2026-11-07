import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from d2nn import config

class FreeSpacePropagation(nn.Module):
    def __init__(self, n: int, dx: float, distance: float, wavelength: float, pad_size: int = 512):
        super().__init__()
        self.n = n
        self.dx = dx
        self.distance = distance
        self.wavelength = wavelength
        self.pad_size = pad_size

        k = 2 * np.pi / wavelength
        f = torch.arange(-pad_size / 2, pad_size / 2, dtype=torch.float32) / (pad_size * dx)
        FX, FY = torch.meshgrid(f, f, indexing="ij")

        arg = (k ** 2) - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2
        mask = arg >= 0
        phase = distance * torch.sqrt(arg * mask)

        self.register_buffer("H", torch.exp(1j * phase) * mask)

    def forward(self, field: torch.Tensor) -> torch.Tensor:

        pad_w = (self.pad_size - self.n) // 2

        field_padded = F.pad(field, (pad_w, pad_w, pad_w, pad_w), mode='constant', value=0)

        F_field = torch.fft.fftshift(torch.fft.fft2(field_padded), dim=(-2, -1))
        F_o = F_field * self.H.to(field.device)
        out_padded = torch.fft.ifft2(torch.fft.ifftshift(F_o, dim=(-2, -1)))

        out = out_padded[..., pad_w:pad_w+self.n, pad_w:pad_w+self.n]
        return out

def center_image_in_canvas(img: torch.Tensor, target_xy_pixel: tuple,
                            canvas_size: int, scale: float = 1) -> torch.Tensor:
    import torch.nn.functional as F
    h0 = w0 = img.shape[-1]
    h1, w1 = int(h0 * scale), int(w0 * scale)
    if (h1, w1) != (h0, w0):
        img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(h1, w1),
                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    canvas = torch.zeros((canvas_size, canvas_size), dtype=img.dtype, device=img.device)
    cx, cy = target_xy_pixel
    half_h, half_w = h1 // 2, w1 // 2
    x1, y1 = cx - half_w, cy - half_h
    x2, y2 = cx + half_w, cy + half_h

    sx1 = max(0, -x1);       sy1 = max(0, -y1)
    sx2 = min(w1, canvas_size - x1); sy2 = min(h1, canvas_size - y1)
    dx1 = max(0, x1);        dy1 = max(0, y1)
    dx2 = min(canvas_size, x2); dy2 = min(canvas_size, y2)
    w = min(sx2 - sx1, dx2 - dx1); h = min(sy2 - sy1, dy2 - dy1)
    if w > 0 and h > 0:
        canvas[dy1:dy1 + h, dx1:dx1 + w] = img[sy1:sy1 + h, sx1:sx1 + w]
    return canvas
