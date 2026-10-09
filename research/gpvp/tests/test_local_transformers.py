from __future__ import annotations

import unittest

from research.gpvp.local_transformers import (
    TransformersLocalAdapter,
    _replace_primary_output,
    find_decoder_layers,
)


class Layer:
    pass


class Nested:
    def __init__(self):
        self.layers = [Layer(), Layer()]


class FakeModel:
    def __init__(self):
        self.model = Nested()


class LocalTransformersTests(unittest.TestCase):
    def test_finds_supported_layer_list(self) -> None:
        path, layers = find_decoder_layers(FakeModel())
        self.assertEqual(path, "model.layers")
        self.assertEqual(len(layers), 2)

    def test_replaces_only_primary_tuple_output(self) -> None:
        marker = object()
        output = ("hidden", "cache", marker)
        updated = _replace_primary_output(output, "changed")
        self.assertEqual(updated, ("changed", "cache", marker))

    def test_local_evidence_requires_immutable_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "40-character"):
            TransformersLocalAdapter("local/model", revision="main")


if __name__ == "__main__":
    unittest.main()
