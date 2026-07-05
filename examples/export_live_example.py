"""Example of exporting a streamable conv-net to a Core ML ``.pte`` for ``neural.live~``.

It demonstrates two extra inputs: an ``attribute`` (``gain``) and a ``condition``
(``bias``), each registered via ``register_method(inputs=[...])``

If the export errors about ``flatc``,
prepend ``<executorch>/cmake-out/third-party/flatc_ep/bin`` to ``PATH``.

Run:

    python export_live_example.py [out_name]

"""

import logging
import sys

import cached_conv as cc
import torch
import torch.nn as nn

from neural_tilde import LiveModule

# cc.Conv1d picks CachedConv1d 
cc.use_cached_conv(True)


class TinyConvNet(LiveModule):

    def __init__(self, hidden: int = 16, kernel_size: int = 3):
        super().__init__()
        pad = cc.get_padding(kernel_size)
        self.net = cc.CachedSequential(
            cc.Conv1d(2, hidden, kernel_size, padding=pad),
            nn.GELU(),
            cc.Conv1d(hidden, 2, kernel_size, padding=pad)
        )

    def forward(self,
                x: torch.Tensor,        # [batch, 2, time]   (signal)
                gain: torch.Tensor,     # [1]                (attribute)
                bias: torch.Tensor      # [batch, 1, 1]         (condition)
                # the argument order should match register_method(inputs=["gain", "bias"]).
                ) -> torch.Tensor:
        x = x + bias.reshape(-1, 1, 1)
        y = self.net(x)                                      # [batch, 2, time]
        return y * gain.reshape(1, 1, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_stream"
    delegate = sys.argv[2] if len(sys.argv) > 2 else "mlx"  # e.g. coreml, xnnpack, mps

    model = TinyConvNet()

    # Extra (non-signal) inputs are registered once by name, 
    # then listed in order via register_method(inputs=[...])
    model.register_attribute("gain", 1.0, minimum=0.0, maximum=2.0,
                             description="output gain")
    model.register_condition("bias", [1, 1], dtype="float32",
                             description="held DC offset added before the net")

    # Registering the method
    model.register_method(
        "forward",
        in_channels=2,
        in_ratio=1,
        out_channels=2,
        out_ratio=1,
        input_labels=["(signal) audio in ch1", "(signal) audio in ch2"],
        output_labels=["(signal) audio out ch1", "(signal) audio out ch2"],
        inputs=["gain", "bias"]
    )

    # Lower with the chosen delegate (default "mlx"; pass a 2nd CLI arg to override,
    # e.g. "xnnpack" for CPU, "coreml" for Apple, "mps" for Apple GPU). For coreml, pass
    # coreml_compute_units="CPU_AND_NE"/"CPU_AND_GPU"/... to pick the unit (default "ALL").
    path = model.export_to_pte(out, delegate=delegate, buffer_size=4096, strict=True)
    print(f"wrote {path} and {path[:-4]}.json")

if __name__ == "__main__":
    main()
