#!/usr/bin/env python3
"""Read-only checkpoint checks inside the final image; no table materialization."""
import hashlib
import json
from pathlib import Path
import struct


def main():
    import importlib.util
    import torch
    assert torch.cuda.get_device_capability() == (12,1), "SM121 required"
    model = Path("/model")
    config = json.loads((model/"config.json").read_text())
    text_config = config.get("text_config",config)
    quant = config["quantization_config"]
    assert (quant["expert_dtype"],quant["engram_dtype"],quant["engram_block_size"],quant["engram_scale_fmt"]) == ("fp4","fp4",32,"ue8m0")
    assert text_config["max_position_embeddings"] == 1048576
    assert text_config["dspark_block_size"] == 5
    assert text_config["engram_layer_ids"] == [1,14]
    index = json.loads((model/"model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 48
    for name in set(index["weight_map"].values()):
        assert (model/name).is_file(), name
    for layer, rows in zip([1,14],text_config["engram_num_embeddings"]):
        key = f"layers.{layer}.engram.embed"
        shard = model/index["weight_map"][key+".weight"]
        with shard.open("rb",buffering=0) as file:
            length = struct.unpack("<Q",file.read(8))[0]
            assert length < 16*2**20
            header = json.loads(file.read(length))
        for suffix,dtype,shape in [("weight","U8",[rows,128]),("scale","F8_E8M0",[rows,8])]:
            tensor = header[key+"."+suffix]
            assert (tensor["dtype"],tensor["shape"]) == (dtype,shape)
            assert tensor["data_offsets"][1]+8+length <= shard.stat().st_size
    here = Path(__file__).resolve().parent
    import os
    launch = json.loads(os.environ.get("DS41_CONFIG_JSON", "{}"))
    if launch.get("performance_patch"):
        manifest = json.loads((here/"patches/performance-hashes.json").read_text())
        base = Path(importlib.util.find_spec("b12x").origin).parent.parent
        for relative, hashes in manifest.items():
            digest = hashlib.sha256(((base/relative).read_text().rstrip()+"\n").encode()).hexdigest()
            assert digest == hashes["output"], f"Performance patch mismatch: {relative}"
    hashes = json.loads((here/"patches/source-hashes.json").read_text())
    for package, relative in [("vllm","vllm/models/deepseek_v4_1/common/engram.py"),
                              ("b12x","b12x/sequence/engram/api.py"),
                              ("b12x","b12x/sequence/engram/_kernels.py")]:
        base = Path(importlib.util.find_spec(package).origin).parent.parent
        digest = hashlib.sha256(((base/relative).read_text().rstrip()+"\n").encode()).hexdigest()
        assert digest == hashes[relative]["output"], f"Patch mismatch: {relative}"
    # Output comparable across hosts without rereading ~384 GiB every launch.
    digest = hashlib.sha256((model/"config.json").read_bytes()+(model/"model.safetensors.index.json").read_bytes()).hexdigest()
    print(f"DS41_PREFLIGHT_PASS config_index_sha256={digest}")


if __name__ == "__main__":
    main()
