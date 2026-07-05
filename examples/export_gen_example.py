"""Minimal example: export a tiny generative model to ``.pte`` for ``neural.gen~``.

This is the simplest possible :class:`GenModule` demo — one method, two inputs:

- ``input_ids``  (condition role) — token ids from a tokenizer or hand-crafted int tensor
- ``x``          (noise role)     — the latent noise the model decodes into audio

Run::

    python export_gen_example_mini.py [out_name]

Writes ``<out_name>.pte`` + ``<out_name>.json`` (default stem: ``tiny_gen``).

If the export errors about ``flatc``, prepend
``<executorch>/cmake-out/third-party/flatc_ep/bin`` to ``PATH``.
"""

import logging
import sys

import torch
import torch.nn as nn

from neural_tilde import GenModule

VOCAB = 32      # toy vocabulary
LATENT = 128      # latent channels
STEPS = 32      # latent time steps
LENGTH = STEPS * STEPS  # 1024 audio samples (STEPS× upsample of the STEPS-step latent)

class TinyGen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, LATENT)
        self.up = nn.Conv1d(LATENT, 2 * STEPS, kernel_size=1) # Upsampler
        self.enc = nn.Conv1d(2, LATENT, kernel_size=LENGTH // STEPS, stride=LENGTH // STEPS)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.up(z)                          # [1, 2*STEPS, STEPS]
        h = h.view(1, 2, STEPS, STEPS)          # [1, C=2, r=STEPS, L=STEPS]
        h = h.permute(0, 1, 3, 2)               # [1, 2, L=STEPS, r=STEPS]
        y = h.reshape(1, 2, STEPS * STEPS)      # [1, 2, LENGTH]
        return torch.tanh(y)

    def line2audio(self,
                     line_ids: torch.Tensor,   # [1, 64]            int64   (condition)
                     noise_x: torch.Tensor,     # [1, LATENT, STEPS] float32 (noise)
                     ) -> torch.Tensor:
        emb = self.embed(line_ids)             # [1, 64, LATENT]
        cond = emb.mean(1)                      # [1, LATENT]
        z = noise_x + cond.unsqueeze(-1).expand(-1, -1, STEPS)  # [1, LATENT, STEPS]
        return self._decode(z)

    def audio2audio(self,
                    init_audio: torch.Tensor,  # [1, 2, LENGTH]     float32 (buffer)
                    noise_x: torch.Tensor,     # [1, LATENT, STEPS] float32 (noise)
                    ) -> torch.Tensor:
        enc = self.enc(init_audio)             # [1, LATENT, STEPS]
        z = noise_x + 0.5 * enc
        return self._decode(z)

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_gen"
    delegate = sys.argv[2] if len(sys.argv) > 2 else "coreml"  # e.g. mps, mlx, xnnpack

    gm = GenModule(TinyGen())
    gm.register_condition("line_ids", [1, 64], "int64")
    gm.register_noise("noise_x", [1, LATENT, STEPS])
    gm.register_method("line2audio",
                       inputs=["line_ids", "noise_x"],
                       out_channels=2, out_length=LENGTH,
                       out_sample_rate=44100, test_method=True)
    gm.register_buffer_input("init_audio", channels=2, length=LENGTH, sample_rate=44100)
    gm.register_method("audio2audio",
                       inputs=["init_audio", "noise_x"],
                       out_channels=2, out_length=LENGTH,
                       out_sample_rate=44100, test_method=True)

    # For the coreml delegate, pass coreml_compute_units="CPU_AND_NE"/"CPU_AND_GPU"/... to
    # pick the Core ML compute unit (default "ALL" — CPU/GPU/ANE, chosen per op by Core ML).
    path = gm.export_to_pte(out, delegate=delegate)
    print(f"wrote {path} and {path[:-4]}.json")
    print("load in Max: [neural.gen~ tiny_gen.pte line2audio]")
    print("         or: [neural.gen~ tiny_gen.pte audio2audio]")


if __name__ == "__main__":
    main()
