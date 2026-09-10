"""Small synthetic containers test index integrity without any model downloads."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "hybrid", Path(__file__).resolve().parents[1] / "scripts/prepare-deepseek-v41-hybrid.py")
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


class HybridIndexTests(unittest.TestCase):
    def shard(self, root, dtype="U8", width=128):
        name = "layers.1.engram.embed.weight"
        data = json.dumps({name: {"dtype": dtype, "shape": [1, width],
                                 "data_offsets": [0, width]}}).encode()
        path = root / "test.safetensors"
        path.write_bytes(struct.pack("<Q", len(data)) + data + bytes(width))
        return path, {name: path.name}

    def test_index_counts_payload_not_container_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, expected = self.shard(root)
            with patch.object(hybrid, "SHARDS", [path.name]):
                result = hybrid.build_index(root, expected)
            self.assertEqual(result["metadata"]["total_size"], 128)
            self.assertEqual(result["weight_map"], expected)

    def test_rejects_fp8_table(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, expected = self.shard(root, "F8_E4M3", 256)
            with patch.object(hybrid, "SHARDS", [path.name]):
                with self.assertRaisesRegex(ValueError, "packed FP4"):
                    hybrid.build_index(root, expected)

    def test_rejects_missing_tensor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, expected = self.shard(root)
            expected["missing.weight"] = path.name
            with patch.object(hybrid, "SHARDS", [path.name]):
                with self.assertRaisesRegex(ValueError, "tensor set"):
                    hybrid.build_index(root, expected)

    def test_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.shard(Path(temp))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                hybrid.header(path)


if __name__ == "__main__":
    unittest.main()
