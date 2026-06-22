"""Minimal example: export a tiny generative model to ``.pte`` for ``neural.gen~``.

This is the simplest possible :class:`GenModule` demo — one method, two inputs:

- ``input_ids``  (condition role) — token ids from a tokenizer or hand-crafted int tensor
- ``x``          (noise role)     — the latent noise the model decodes into audio

Run::

    python export_gen_example_mini.py [out_stem]

Writes ``<out_stem>.pte`` + ``<out_stem>.json`` (default stem: ``tiny_gen``).

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
LENGTH = STEPS * STEPS  # 256 audio samples (16× upsample of the 16-step latent)

class TinyGen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, LATENT)
        self.up = nn.ConvTranspose1d(LATENT, 2, kernel_size=STEPS, stride=STEPS)
        self.enc = nn.Conv1d(2, LATENT, kernel_size=LENGTH // STEPS, stride=LENGTH // STEPS)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.up(z))           # [1, 2, LENGTH]

    def prompt2audio(self,
                     vocab_ids: torch.Tensor,   # [1, 64]            int64   (condition)
                     noise_x: torch.Tensor,     # [1, LATENT, STEPS] float32 (noise)
                     ) -> torch.Tensor:
        emb = self.embed(vocab_ids)             # [1, 64, LATENT]
        cond = emb.mean(1)                      # [1, LATENT]
        z = noise_x + cond.unsqueeze(-1)        # [1, LATENT, STEPS]
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

    gm = GenModule(TinyGen())
    gm.register_condition("vocab_ids", [1, 64], "int64")
    gm.register_noise("noise_x", [1, LATENT, STEPS])
    gm.register_method("prompt2audio",
                       inputs=["vocab_ids", "noise_x"],
                       out_channels=2, out_length=LENGTH,
                       out_sample_rate=44100, test_method=True)
    gm.register_buffer_input("init_audio", channels=2, length=LENGTH, sample_rate=44100)
    gm.register_method("audio2audio",
                       inputs=["init_audio", "noise_x"],
                       out_channels=2, out_length=LENGTH,
                       out_sample_rate=44100, test_method=True)

    path = gm.export_to_pte(out, delegate="coreml")
    print(f"wrote {path} and {path[:-4]}.json")
    print("load in Max: [neural.gen~ tiny_gen.pte prompt2audio]")
    print("         or: [neural.gen~ tiny_gen.pte audio2audio]")


if __name__ == "__main__":
    main()
