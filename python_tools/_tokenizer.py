"""``neural_tilde.Tokenizer`` — a reusable tokenizer-bundle descriptor.

A model that takes a text prompt feeds it as token ``condition`` inputs (ids +
attention mask). The text-to-tokens step itself is **not** done by this package nor
by the model ``.pte``: it is performed by the Max ``neural.tokenizer`` object (or
the headless ``gen_cli``) using a HuggingFace ``tokenizer.json`` plus a small
settings file. This class only *describes and bundles* that pair — it does not
tokenize anything.

On export the bundle is written as two standalone files beside the ``.pte``
(``<stem>.tokenizer.json`` + ``<stem>.tokenizer.config.json``) and is **independent
of the model metadata and of the model kind**: the host discovers it by filename
convention and the runtime sees only the resulting token ``condition`` inputs
(matched by name). So either exporter — :class:`LiveModule` or :class:`GenModule` —
may ship one; see ``EXECUTORCH_PROTOCOL.md`` §3.1 / §7 and the C++ ``TokenizerConfig``
struct in ``src/tokenizer/tokenizer_config.h``.

The config's ``ids_key`` / ``mask_key`` MUST equal the token-condition input names
of whichever method consumes the tokens, so the dictionary ``neural.tokenizer``
emits lines up with the model's inputs.

Usually constructed for you by ``register_tokenizer(...)`` on either exporter, but
also usable directly::

    tok = Tokenizer("tokenizer.json", max_length=256,
                    ids_key="input_ids", mask_key="attention_mask")
"""

import json
import logging
import os
import shutil


class Tokenizer:
    """Describes a HuggingFace tokenizer + its host-side settings, and writes the
    standalone ``<stem>.tokenizer.json`` + ``<stem>.tokenizer.config.json`` bundle.
    """

    def __init__(self,
                 tokenizer_file: str,
                 max_length: int,
                 pad_token: str = "<pad>",
                 padding_side: str = "right",
                 ids_key: str = "input_ids",
                 mask_key: str = "attention_mask") -> None:
        """
        Args:
            tokenizer_file: path to a HuggingFace ``tokenizer.json`` (copied on export).
            max_length: fixed sequence length the host pads/truncates to.
            pad_token: padding token literal.
            padding_side: ``"right"`` or ``"left"``.
            ids_key: dict key for the token ids (= the model's token-ids input name).
            mask_key: dict key for the attention mask (= the model's mask input name).
        """
        self.tokenizer_file = tokenizer_file
        self.max_length = int(max_length)
        self.pad_token = pad_token
        self.padding_side = padding_side
        self.ids_key = ids_key
        self.mask_key = mask_key

    def config_dict(self, tok_filename: str) -> dict:
        """The ``<stem>.tokenizer.config.json`` body referencing ``tok_filename``
        (the bundled HF tokenizer.json, relative to the config). Keys/order match the
        C++ ``TokenizerConfig`` struct (``src/tokenizer/tokenizer_config.h``)."""
        return {
            "type": "tokenizers-cpp",
            "tokenizer_file": tok_filename,        # relative to this config file
            "max_length": self.max_length,
            "padding": "max_length",
            "pad_token": self.pad_token,
            "padding_side": self.padding_side,
            "ids_key": self.ids_key,
            "mask_key": self.mask_key,
        }

    def write_files(self, pte_path: str) -> None:
        """Copy the HF tokenizer.json to ``<stem>.tokenizer.json`` and write the
        standalone ``<stem>.tokenizer.config.json`` beside ``pte_path``.

        ``pte_path`` may be the model ``.pte`` path or a bare stem: a trailing
        ``.pte`` is stripped if present, otherwise the whole basename is the stem
        (so a name with dots, e.g. ``sa3.small``, is kept intact)."""
        base = os.path.basename(pte_path)
        stem = base[:-len(".pte")] if base.endswith(".pte") else base
        out_dir = os.path.dirname(os.path.abspath(pte_path))
        tok_filename = f"{stem}.tokenizer.json"
        shutil.copyfile(self.tokenizer_file, os.path.join(out_dir, tok_filename))
        config_path = os.path.join(out_dir, f"{stem}.tokenizer.config.json")
        with open(config_path, "w") as f:
            json.dump(self.config_dict(tok_filename), f, indent=2)
        logging.info(f"Wrote tokenizer bundle {tok_filename} + {os.path.basename(config_path)}")
