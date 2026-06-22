"""Minimal example: bundle a HuggingFace tokenizer for ``neural.tokenizer`` in Max.

No model or export needed — just wrap an existing ``tokenizer.json`` into the two
sidecar files that ``neural.tokenizer`` loads.

Run::

    python export_tokenizer_example.py tokenizer.json [out_stem]

Writes::

    <out_stem>.tokenizer.json          — copy of the HF tokenizer
    <out_stem>.tokenizer.config.json   — settings (max_length, keys, …)

Default ``out_stem`` is ``my_tokenizer``.
"""

import sys

from neural_tilde import Tokenizer


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: export_tokenizer_example.py tokenizer.json [out_stem]")
        sys.exit(1)

    tokenizer_file = sys.argv[1]
    out_stem = sys.argv[2] if len(sys.argv) > 2 else "my_tokenizer"

    tok = Tokenizer(tokenizer_file, max_length=256,
                    ids_key="input_ids", mask_key="attention_mask")
    tok.write_files(out_stem)

    print(f"wrote {out_stem}.tokenizer.json")
    print(f"      {out_stem}.tokenizer.config.json")
    print(f"load in Max: [neural.tokenizer {out_stem}.tokenizer.config.json]")


if __name__ == "__main__":
    main()
