import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import serve
import cluster


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"cluster.json").read_text())

    def test_rank_command_keeps_upstream_precision_and_cluster_contract(self):
        for rank in range(4):
            cmd = serve.command(self.config,rank)
            self.assertEqual(cmd[cmd.index("--node-rank")+1],str(rank))
            self.assertEqual("--headless" in cmd,rank != 0)
            self.assertEqual(cmd[cmd.index("--master-addr")+1],"192.168.0.1")
            self.assertNotIn("--enforce-eager",cmd)
            self.assertNotIn("--quantization",cmd)
            self.assertNotIn("--kv-cache-dtype",cmd)
            self.assertEqual(cmd[cmd.index("--max-model-len")+1],"1048576")
            self.assertEqual(cmd[cmd.index("--max-num-seqs")+1],"16")

    def test_graph_capacity_covers_fixed_dspark_decode_only(self):
        cmd = serve.command(self.config,0)
        cfg = json.loads(cmd[cmd.index("--compilation-config")+1])
        spec = json.loads(cmd[cmd.index("--speculative-config")+1])
        self.assertEqual(cfg["cudagraph_mode"],"FULL_DECODE_ONLY")
        self.assertEqual(cfg["cudagraph_capture_sizes"],list(range(6,97,6)))
        self.assertEqual(cfg["mode"],0)
        self.assertFalse(cfg["pass_config"]["fuse_allreduce_rms"])
        self.assertEqual(max(cfg["cudagraph_capture_sizes"]),16*(spec["num_speculative_tokens"]+1))
        self.assertEqual(spec["method"],"dspark")
        self.assertFalse(spec["enable_adaptive_verification"])

    def test_fabric_uses_dual_rail_roce_not_pcie(self):
        env = cluster.environment(self.config,2,"test-cx0")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"],"test-cx0")
        self.assertEqual(env["NCCL_IB_HCA"],"rocep1s0f0,roceP2p1s0f0")
        self.assertEqual(env["VLLM_HOST_IP"],"192.168.0.3")
        self.assertEqual(env["VLLM_ENABLE_PCIE_ALLREDUCE"],"0")
        self.assertEqual(env["VLLM_USE_BREAKABLE_CUDAGRAPH"],"0")
        self.assertFalse(any("EXL3" in key or "DCP" in key for key in env))

    def test_fp4_encoding_matches_independent_e4m3_decoder(self):
        values = [0.,.5,1,1.5,2,3,4,6]
        for code in range(16):
            magnitude = code&7
            bits = (0 if magnitude==0 else 48 if magnitude==1 else 48+4*magnitude) | ((code&8)<<4)
            exponent, mantissa = (bits>>3)&15,bits&7
            decoded = (mantissa/8)*2**(-6) if exponent==0 else (1+mantissa/8)*2**(exponent-7)
            if bits&128:
                decoded = -decoded
            expected = -values[magnitude] if code&8 else values[magnitude]
            # struct distinguishes signed zeros as the GPU byte-oracle test does.
            self.assertEqual(struct.pack('f',decoded),struct.pack('f',float(expected)))

    def test_fused_fp4_magnitudes_and_sign(self):
        values = [0.,.5,1.,1.5,2.,3.,4.,6.]
        for byte in range(256):
            for shift in (0,4):
                code=(byte>>shift)&15
                mag=code&7
                decoded=mag*.5 if mag<4 else (2+(mag&1))*(1. if mag<6 else 2.)
                if code&8:
                    decoded=-decoded
                expected=-values[mag] if code&8 else values[mag]
                self.assertEqual(struct.pack('f',decoded),struct.pack('f',expected))

    def test_invalid_topology_rejected(self):
        self.config["decode_context_parallel"] = 4
        with self.assertRaises(ValueError):
            serve.command(self.config,0)

    def test_cuda_disk_options_and_model_readonly(self):
        cmd = cluster.docker_base(self.config,0,"cx0")
        self.assertIn("seccomp=unconfined",cmd)
        self.assertIn(self.config["model_path"]+":/model:ro",cmd)
        self.assertNotIn("--privileged",cmd)


if __name__ == "__main__":
    unittest.main()
