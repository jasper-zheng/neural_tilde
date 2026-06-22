"""neural_tilde — export PyTorch neural-audio models to ExecuTorch ``.pte`` +
sidecar JSON for the ``neural.live~`` / ``neural.gen~`` Max externals.

Both exporters register the model's extra inputs by name (attributes / noise / conditions,
plus gen's init buffer), then declare one or more ``register_method`` generation/processing
paths — each listing, in ``forward`` order, the inputs it consumes — and ``export_to_pte``.

Public API:
- :class:`LiveModule` (in ``neural_tilde.live_module``) — the audio-rate exporter for
  ``neural.live~``: ``register_method`` each streaming method (one ``signal`` + extra inputs),
  then ``export_to_pte``.
- :class:`GenModule` (in ``neural_tilde.gen_module``) — the offline-generator exporter for
  ``neural.gen~``: ``register_method`` each generation path (its typed inputs + audio output),
  with per-model ``set_seed`` / tokenizer, then ``export_to_pte``.
- :class:`Tokenizer` (in ``neural_tilde._tokenizer``) — the reusable tokenizer-bundle
  descriptor. ``register_tokenizer(...)`` is available on **both** exporters (a text prompt
  arrives as token ``condition`` inputs, so a ``neural.live~`` model may use one too); the
  bundle is written standalone beside the ``.pte`` and is independent of the model kind.

See ``EXECUTORCH_PROTOCOL.md``.
"""

from ._tokenizer import Tokenizer
from .gen_module import GenModule
from .live_module import LiveModule

__all__ = ["LiveModule", "GenModule", "Tokenizer"]
