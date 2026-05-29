"""Vendored reaction-model architecture.

These modules (``tokenizer``, ``transformer``, ``decoder_model``) are vendored from
the reaction-model training repository so that RxnMol's ``transformer_v2`` adapter is
self-contained — no external ``repo_path`` is required to run fragment mode. They are
imported lazily by :mod:`rxnmol.solutions.reaction_models.transformer_v2` when a model
is instantiated.
"""
