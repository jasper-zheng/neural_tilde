"""Example of exporting a *stateful* conv-net to a Core ML ``.pte`` for ``neural.live~``.

Alongside the audio signal it also demonstrates two extra inputs: a scalar ``attribute``
(``gain``, exposed as a Max attribute the host feeds every block) and a held ``condition``
(``bias``, a control value the patch supplies) — each registered once by name and listed,
in order, via ``register_method(inputs=[...])`` so the ``forward`` receives them after ``x``.

Click-Free Streaming
--------------
This model uses ``cached_conv`` (``cc.Conv1d``) — *one* way to build a stateful streaming net: each
layer keeps the previous block's tail so consecutive blocks join without clicks. Any mechanism that
carries internal state across ``execute()`` calls works; the protocol does not mandate ``cached_conv``.
When exporting with ``delegate="coreml"``, the Core ML partitioner takes the model's mutable buffers
over as native Core ML *state* (``take_over_mutable_buffer=True``), so the state persists across
``execute()`` calls.

If the export errors about ``flatc``,
prepend ``<executorch>/cmake-out/third-party/flatc_ep/bin`` to ``PATH``.

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

    model = TinyConvNet()

    # Extra (non-signal) inputs are registered once by name, 
    # then listed in order via register_method(inputs=[...])
    model.register_attribute("gain", 1.0, minimum=0.0, maximum=2.0,
                             description="output gain")
    model.register_condition("bias", [1, 1], dtype="float32",
                             description="held DC offset added before the net")

    # Registering the method validates the tensor signature AND (test_method=True) runs
    # one forward per batch mode, which also lazily allocates cached_conv's caches.
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

    # Lower to Core ML (runs on the Apple Neural Engine). Swap to
    # delegate="xnnpack" for a CPU build that also streams. The cached_conv
    # caches persist across blocks (taken over as native Core ML state), so the
    # model streams click-free — no sidecar flag needed.
    path = model.export_to_pte(out, delegate="coreml", buffer_size=4096, strict=True)
    print(f"wrote {path} and {path[:-4]}.json")

if __name__ == "__main__":
    main()
