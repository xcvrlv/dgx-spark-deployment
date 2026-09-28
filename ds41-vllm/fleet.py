#!/usr/bin/env python3
"""Run on Spark 1. Standard-library SSH/Docker launcher for four Spark nodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
NAME = 'ds41-jj'
MODEL = 'DeepSeek-V4.1-Flash'
ROCE_OPTIONS = {
    'inline_payload': 'B12X_ROCE_INLINE_PAYLOAD',
    'balanced_fanout': 'B12X_ROCE_BALANCED_FANOUT',
    'skip_empty_cq': 'B12X_ROCE_SKIP_EMPTY_CQ',
    'lazy_payload_init': 'B12X_ROCE_LAZY_PAYLOAD_INIT',
}
DSPARK_PREFILL_OPTIONS = {
    'dspark_skip_prefill_draft': 'DS41_SKIP_PREFILL_DRAFT',
    'dspark_compact_context_graph': 'DS41_COMPACT_CONTEXT_GRAPH',
}
LATEST_PINS = ('502d6cb5acd2ba2a62ecf58497be558c9d86089f',
               'd44247b6171f7c2f9787341ae884b537887d7df9')
OVERLAY_PINS = {
    ('1794dcf18454900263e0c66711af8ea4a1283ac1',
     'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'), LATEST_PINS,
}
PERFORMANCE_ENV = {
    'engram_overlap': 'VLLM_DS41_ENGRAM_OVERLAP',
    'dspark_markov_nvfp4': 'VLLM_DS41_MARKOV_NVFP4',
    'dspark_draft_nvfp4_head': 'VLLM_DS41_DRAFT_NVFP4_HEAD',
    'l2_prefetch': 'VLLM_DS41_L2_PREFETCH',
    'h2d_staging': 'DS41_H2D_STAGING',
    'b12x_defer_gc': 'DS41_DEFER_AUTOTUNE_GC',
    'bounded_prefix_hashes': 'DS41_BOUNDED_PREFIX_HASHES',
    'moe_coalesce_barriers': 'DS41_MOE_COALESCE_BARRIERS',
}


def source_pins(c):
    return c.get('vllm_commit'), c.get('b12x_commit')


def exact_decode_graph_sizes(c):
    # Include draft query width K and every adaptive target verification depth
    # 1..K+1 for each admitted request count. No depth relies on an eager gap.
    return sorted({requests * width
                   for requests in range(1, c['max_num_seqs'] + 1)
                   for width in range(1, c['draft_tokens'] + 2)})
# The display reserve is firmware memory the OS cannot use, so it is credited to
# the KV budget rather than reached through gpu_memory_utilization. The DRM group
# is resolved on the node; this token is shell-expanded in plan(), not quoted.
DISPLAY_KV_GID_NAME = 'DS41_DISPLAY_KV_GID'
DISPLAY_KV_GID = '$' + DISPLAY_KV_GID_NAME
DISPLAY_KV_MAX_MIB = 1792


def display_kv(c):
    """Display-reserve credit for this deployment, or None when disabled."""
    settings = c.get('display_kv', {})
    return settings if settings.get('enabled', False) else None


def load_config(path):
    c = json.loads(Path(path).read_text())
    assert len(c['nodes']) == 4
    assert len({n['host'] for n in c['nodes']}) == 4
    assert len({n['ip'] for n in c['nodes']}) == 4
    assert len(c['hcas']) == 2
    assert type(c.get('reduced_tuning', True)) is bool
    assert type(c.get('b12x_autotune', False)) is bool
    assert type(c.get('enable_prompt_tokens_details', True)) is bool
    assert type(c.get('b12x_bounded_autotune', False)) is bool
    assert type(c.get('prefill_8192_graph', False)) is bool
    if c.get('prefill_8192_graph', False):
        assert c.get('upstream_branch') == 'dev/karmic-kraken'
        assert source_pins(c) in OVERLAY_PINS
        assert c['max_num_batched_tokens'] == 8192
        if source_pins(c) != LATEST_PINS:
            assert c.get('b12x_bounded_autotune') is True
    if c.get('b12x_bounded_autotune', False):
        assert c.get('upstream_branch') == 'dev/karmic-kraken'
        assert source_pins(c) in OVERLAY_PINS
        assert c.get('b12x_autotune') is True
        assert type(c.get('b12x_compile_workers')) is int and 1 <= c['b12x_compile_workers'] <= 2
        assert c.get('b12x_preparation_trace') is True and c.get('b12x_hang_dump') is True
        for key, default, low, high in (
            ('b12x_race_budget_mib', 1024, 64, 4096),
            ('b12x_memory_reserve_mib', 4096, 4096, 16384),
        ):
            value = c.get(key, default)
            assert type(value) is int and low <= value <= high, f'Invalid {key}'
    assert type(c.get('torch_profile', False)) is bool
    for key in PERFORMANCE_ENV:
        if key in c:
            assert type(c[key]) is bool, f'{key} must be a boolean'
            assert source_pins(c) == LATEST_PINS, f'{key} requires the audited latest pins'
    for key, choices in (('draft_sample_method', ('greedy', 'probabilistic')),
                         ('rejection_sample_method', ('standard', 'block')),
                         ('decode_graph_policy', ('upstream', 'exact'))):
        if key in c:
            assert c[key] in choices, f'Invalid {key}'
            assert source_pins(c) == LATEST_PINS
    assert type(c.get('enable_adaptive_verification', True)) is bool
    if 'engram_projection_tp' in c:
        assert type(c['engram_projection_tp']) is bool
        assert source_pins(c) == LATEST_PINS
    if 'adaptive_verification_cost_scale' in c:
        value = c['adaptive_verification_cost_scale']
        assert type(value) in (int, float) and 0 < value < float('inf')
        assert c['draft_tokens'] > 0 and c.get('enable_adaptive_verification', True)
        assert source_pins(c) == LATEST_PINS
    if 'shm_busy_loop_s' in c:
        assert type(c['shm_busy_loop_s']) in (int, float) and 0 <= c['shm_busy_loop_s'] <= 1
        assert source_pins(c) == LATEST_PINS
    if c.get('performance_bundle') is not None:
        assert c['performance_bundle'] == 'ds41-performance-v1'
        assert source_pins(c) == LATEST_PINS
    for key in ('dspark_markov_nvfp4', 'dspark_draft_nvfp4_head'):
        assert not c.get(key, False) or c['draft_tokens'] > 0, f'{key} requires DSpark'
    for key in ('engram_resident_scales', 'graph_memory_debug', *DSPARK_PREFILL_OPTIONS):
        assert type(c.get(key, False)) is bool, f'{key} must be a boolean'
        if c.get(key, False):
            assert c.get('upstream_branch') == 'dev/karmic-kraken', f'{key} requires the audited Karmic profile'
    if any(c.get(key, False) for key in DSPARK_PREFILL_OPTIONS):
        assert c['draft_tokens'] > 0, 'DSpark prefill optimizations require speculative decoding'
        assert source_pins(c) in OVERLAY_PINS, 'DSpark prefill overlay requires audited paired pins'
    if c.get('dspark_compact_context_graph', False):
        assert c['max_num_batched_tokens'] >= 128, 'Compact context graph requires a 128-row buffer'
    assert c.get('upstream_branch', 'dev/jovian-judgement') in (
        'dev/jovian-judgement', 'dev/karmic-kraken')
    if c.get('upstream_branch') == 'dev/karmic-kraken':
        assert type(c.get('b12x_compile_workers')) is int and 1 <= c['b12x_compile_workers'] <= 16
        assert type(c.get('b12x_preparation_trace', False)) is bool
        assert type(c.get('b12x_hang_dump', False)) is bool
        for key in ('vllm_commit', 'b12x_commit'):
            assert isinstance(c.get(key), str) and len(c[key]) == 40
        assert not display_kv(c), 'Karmic image has no display-KV overlay'
        assert not c.get('reduced_tuning', False), 'Karmic image has no tuning overlay'
        assert 'graph_request_buckets' not in c
    assert c['max_num_seqs'] in (8, 16), 'Supported profiles: c8 and c16'
    assert set(c.get('roce_optimizations', {})) <= set(ROCE_OPTIONS)
    assert all(type(v) is bool for v in c.get('roce_optimizations', {}).values())
    assert 0 < c['gpu_memory_utilization'] < 1
    assert 0 < c['max_model_len'] <= 1048576
    assert type(c.get('omp_num_threads', 2)) is int and c.get('omp_num_threads', 2) > 0
    assert c['draft_tokens'] in (0, 1, 3, 5, 7)
    window = c.get('adaptive_speculative_tokens_window')
    initial = c.get('adaptive_speculative_tokens_initial')
    if window is not None:
        assert type(window) is int and window > 0 and c['draft_tokens'] > 0, 'Invalid adaptive window'
    if initial is not None:
        assert window is not None and type(initial) is int and 1 <= initial <= c['draft_tokens'], 'Invalid adaptive initial depth'
    assert c['max_num_batched_tokens'] >= c['max_num_seqs'] * (1 + 2 * c['draft_tokens']), \
        'Batch capacity must cover DSpark parallel-drafting profiling rows'
    for key in ('model_path', 'cache_path'):
        assert c[key].startswith('/') and c[key] != '/' and ',' not in c[key]
    settings = c.get('display_kv', {})
    assert set(settings) <= {'enabled', 'display_mib', 'drm_card'}, 'Unknown display_kv key'
    assert type(settings.get('enabled', False)) is bool
    if settings:
        # Fail closed on a bad credit: the backing span is fixed at this size,
        # so a larger credit would be admitted and then fail at allocation.
        mib = settings.get('display_mib', 0)
        assert type(mib) is int and 1 <= mib <= DISPLAY_KV_MAX_MIB, 'display_mib must be 1..1792'
        card = settings.get('drm_card', '')
        assert type(card) is str and card.startswith('/dev/dri/card'), 'drm_card must be a /dev/dri/card node'
    checkpoint_path(c, '/checkpoint')
    return c


def checkpoint_path(c, root):
    # Keep the whole HF cache mounted for blob symlinks; '.' selects a flat model.
    relative = PurePosixPath(c.get('model_subpath', f"snapshots/{c['revision']}"))
    assert not relative.is_absolute() and '..' not in relative.parts, 'model_subpath must stay inside model_path'
    return str(PurePosixPath(root) / relative)


def ssh(c, rank):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    if c.get('ssh_identity'):
        args += ['-i', c['ssh_identity']]
    return args + [f"{c['ssh_user']}@{c['nodes'][rank]['host']}"]


def remote(c, rank, script, timeout=120):
    prologue = '''set -Eeuo pipefail
trap 'rc=$?; printf "FAILED (exit %s) at remote line %s: %s\\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2' ERR
'''
    proc = subprocess.run(ssh(c, rank) + ['bash', '-s'], input=prologue + script,
                          text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"rank {rank} ({c['nodes'][rank]['host']}), exit {proc.returncode}:\n"
                           f"{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def environment(c, rank):
    environment = {
        **{env: str(int(c.get('roce_optimizations', {}).get(key, False)))
           for key, env in ROCE_OPTIONS.items()},
        'VLLM_HOST_IP': c['nodes'][rank]['ip'],
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_USE_V2_MODEL_RUNNER': '1',
        'VLLM_ENABLE_ROCE_ALLREDUCE': '1', 'VLLM_ENABLE_PCIE_ALLREDUCE': '0',
        'VLLM_ALLREDUCE_USE_SYMM_MEM': '0', 'VLLM_ALLREDUCE_USE_FLASHINFER': '0',
        'VLLM_ROCE_ALLREDUCE_MAX_SIZE': '2MB', 'VLLM_ROCE_ALLGATHER_MAX_SIZE': '16MB',
        'B12X_ROCE_HCA': ','.join(c['hcas']), 'B12X_ROCE_GID_INDEX': str(c['gid_index']),
        'NCCL_NET': 'IB', 'NCCL_IB_DISABLE': '0', 'NCCL_DEBUG': 'INFO',
        'NCCL_IB_HCA': '=' + ','.join(c['hcas']), 'NCCL_IB_GID_INDEX': str(c['gid_index']),
        'NCCL_NVLS_ENABLE': '0', 'NCCL_IB_MERGE_NICS': '0', 'NCCL_CROSS_NIC': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_MIN_NCHANNELS': '4', 'NCCL_MAX_NCHANNELS': '4',
        'NCCL_IGNORE_CPU_AFFINITY': '1',
        'CUTE_DSL_ARCH': 'sm_121a', 'TORCH_CUDA_ARCH_LIST': '12.1a',
        'CUDA_DEVICE_MAX_CONNECTIONS': '32', 'OMP_NUM_THREADS': str(c.get('omp_num_threads', 2)),
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'MALLOC_ARENA_MAX': '2', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'XDG_CACHE_HOME': '/cache',
    }
    if c.get('upstream_branch') == 'dev/karmic-kraken':
        # Compilation workers share physical RAM with the GPU on GB10.
        environment['B12X_COMPILE_WORKERS'] = str(c['b12x_compile_workers'])
        environment['DS41_B12X_BOUNDED_AUTOTUNE'] = str(int(c.get('b12x_bounded_autotune', False)))
        environment['DS41_PREFILL_8192_GRAPH'] = str(int(c.get('prefill_8192_graph', False)))
        if source_pins(c) == LATEST_PINS:
            for key, env in PERFORMANCE_ENV.items():
                # An absent upstream control preserves its upstream default.
                # Local source overlays default off, including the control recipe.
                if key in c or key in ('h2d_staging', 'b12x_defer_gc', 'bounded_prefix_hashes',
                                      'moe_coalesce_barriers'):
                    environment[env] = str(int(c.get(key, False)))
            environment['DS41_SHM_BUSY_LOOP_S'] = str(c.get('shm_busy_loop_s', 1))
        if c.get('b12x_bounded_autotune', False):
            environment.update({
                'B12X_AUTOTUNE': '1',
                'DS41_B12X_RACE_BUDGET_MIB': str(c.get('b12x_race_budget_mib', 1024)),
                'DS41_B12X_RESERVE_MIB': str(c.get('b12x_memory_reserve_mib', 4096)),
                **{f'B12X_{stage}_COMPILE_WORKERS': str(c['b12x_compile_workers'])
                   for stage in ('WEIGHTS', 'STATE', 'BIND')},
            })
        # Explicit zeros let either optimization be rolled back independently.
        environment.update({env: str(int(c.get(key, False)))
                            for key, env in DSPARK_PREFILL_OPTIONS.items()})
        if c.get('b12x_preparation_trace', False):
            environment['B12X_PREPARATION_TRACE_DIR'] = '/cache/b12x-preparation-trace'
        if c.get('b12x_hang_dump', False):
            environment['B12X_HANG_DUMP'] = '1'
        if c.get('graph_memory_debug', False):
            environment['VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING'] = '1'
    else:
        environment['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '0'
    display = display_kv(c)
    if display:
        # Additive only: the display capability set and the credit are absent
        # when the deployment is disabled, so it stays byte-identical to today.
        environment.update(
            NVIDIA_DRIVER_CAPABILITIES='compute,utility,graphics,display',
            DS41_DISPLAY_KV_MIB=str(display['display_mib']),
        )
    return environment


def docker(c, rank, name=None):
    cmd = ['docker', 'run', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
           '--ulimit', 'memlock=-1:-1', '--ulimit', 'nofile=1048576:1048576',
           '--device', '/dev/infiniband:/dev/infiniband',
           '--security-opt', 'seccomp=unconfined',  # io_uring is blocked by Docker's default profile
           '--mount', f"type=bind,src={c['model_path']},dst=/checkpoint,readonly",
           '--mount', f"type=bind,src={c['cache_path']},dst=/cache",
           '--entrypoint', '', '-e', 'NCCL_SOCKET_IFNAME', '-e', 'GLOO_SOCKET_IFNAME']
    display = display_kv(c)
    if display:
        # Expose only the DRM card and its group, never elevate the worker.
        # The group id is resolved on the node and expanded by plan().
        cmd += ['--device', display['drm_card'], '--group-add', DISPLAY_KV_GID]
        cmd += ['-e', DISPLAY_KV_GID_NAME]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for key, value in environment(c, rank).items():
        cmd += ['-e', f'{key}={value}']
    return cmd + [c['image']]


def setup(c, rank):
    # The interface is discovered separately on every node, never copied from rank 0.
    ip = shlex.quote(c['nodes'][rank]['ip'])
    script = f'''iface=$(ip -o -4 address show | awk -v wanted={ip} 'split($4,a,"/") && a[1]==wanted {{print $2}}')
test -n "$iface"
test "$(printf '%s\\n' "$iface" | wc -l)" = 1
export NCCL_SOCKET_IFNAME="=$iface" GLOO_SOCKET_IFNAME="$iface"
'''
    display = display_kv(c)
    if display:
        # The DRM group is discovered on the node for the same reason: it is a
        # node property, so it is never copied from rank 0.
        card = shlex.quote(display['drm_card'])
        script += f'''test -c {card}
export {DISPLAY_KV_GID_NAME}="$(stat -c '%g' {card})"
'''
    return script


def serve_args(c, rank):
    engram = {'cpu_offload': False, 'table_memory': 'disk'}
    if c.get('engram_resident_scales', False):
        # Upstream retains exact E8M0 bytes: about 1.43 GiB/rank at TP4.
        # Opt-in only; this competes with KV and graphs in GB10's shared RAM.
        engram['disk_resident_scales'] = True
    if 'engram_projection_tp' in c:
        engram['projection_tp'] = c['engram_projection_tp']
    cmd = ['vllm', 'serve', checkpoint_path(c, '/checkpoint'),
           '--served-model-name', MODEL, '--host', '0.0.0.0', '--port', str(c['port']),
           '--distributed-executor-backend', 'mp', '--nnodes', '4', '--node-rank', str(rank),
           '--master-addr', c['nodes'][0]['ip'], '--master-port', str(c['master_port']),
           '--tensor-parallel-size', '4', '--decode-context-parallel-size', '1',
           '--dtype', 'bfloat16', '--load-format', 'safetensors', '--safetensors-load-strategy', 'lazy',
           '--attention-backend', 'B12X', '--linear-backend', 'b12x', '--moe-backend', 'b12x',
           '--block-size', '256', '--kv-cache-dtype', 'fp8',
           '--engram-config', json.dumps(engram),
           '--gpu-memory-utilization', str(c['gpu_memory_utilization']),
           '--max-model-len', str(c['max_model_len']), '--max-num-seqs', str(c['max_num_seqs']),
           '--max-num-batched-tokens', str(c['max_num_batched_tokens']),
           '--enable-prefix-caching', '--enable-chunked-prefill', '--async-scheduling',
           '--no-scheduler-reserve-full-isl',
           '--generation-config', 'vllm', '--reasoning-parser', 'deepseek_v41',
           '--tool-call-parser', 'deepseek_v41', '--enable-auto-tool-choice']
    if c.get('enable_prompt_tokens_details', True):
        cmd += ['--enable-prompt-tokens-details']
    if c.get('upstream_branch') != 'dev/karmic-kraken':
        depth = c['draft_tokens'] + 1
        maximum = c['max_num_seqs'] * depth
        # Legacy JJ capture spread; Karmic Kraken uses upstream defaults.
        sizes = sorted({size for size in (1, 2, 8) if size <= maximum} | {maximum})
        compilation = {'cudagraph_mode': 'FULL_AND_PIECEWISE', 'custom_ops': ['all'],
                       'cudagraph_capture_sizes': sizes,
                       'pass_config': {'fuse_allreduce_rms': False}}
        cmd += ['--compilation-config', json.dumps(compilation)]
    elif c.get('decode_graph_policy') == 'exact':
        cmd += ['--compilation-config', json.dumps({
            'cudagraph_mode': 'FULL_AND_PIECEWISE',
            'cudagraph_capture_sizes': exact_decode_graph_sizes(c),
        })]
    if not c.get('b12x_autotune', False) or c.get('b12x_bounded_autotune', False):
        # Startup candidate racing overdrafts the device on this fleet. With
        # autotune off nothing is timed: every choice is prepared with its
        # default or cached configuration. The per-field backend flags above
        # still apply on top of this JSON in create_engine_config.
        cmd += ['--kernel-config', json.dumps({'enable_b12x_autotune': c.get('b12x_autotune', False)})]
    if c['draft_tokens']:
        cmd += ['--speculative-config', json.dumps({
            'method': 'dspark', 'num_speculative_tokens': c['draft_tokens'],
            'draft_tensor_parallel_size': 4, 'attention_backend': 'B12X',
            'draft_sample_method': c.get('draft_sample_method', 'greedy'),
            'rejection_sample_method': c.get('rejection_sample_method', 'standard'),
            'enable_adaptive_verification': c.get('enable_adaptive_verification', True),
            **{key: c[key] for key in (
                'adaptive_speculative_tokens_window', 'adaptive_speculative_tokens_initial',
                'adaptive_verification_cost_scale'
            ) if c.get(key) is not None}})]
    if c.get('swa_block_size') is not None:
        cmd += ['--swa-block-size', str(c['swa_block_size'])]
    # No new vllm flag carries the credit: it travels as DS41_DISPLAY_KV_MIB in
    # environment() and is read by the patched KV path, so stock vllm never has
    # to understand a flag it does not define.
    if c.get('torch_profile', False):
        cmd += ['--profiler-config', json.dumps({
            'profiler': 'torch', 'torch_profiler_dir': '/cache/profiles',
            'torch_profiler_with_stack': False,
            'torch_profiler_record_shapes': True,
            'torch_profiler_with_memory': False})]
    if rank:
        cmd += ['--headless']
    return cmd


def plan(c, rank):
    line = shlex.join(docker(c, rank, f'{NAME}-{rank}') + serve_args(c, rank))
    if display_kv(c):
        # The DRM group is resolved on the node, so this one token has to be
        # expanded by the shell instead of quoted as a literal by shlex.join.
        line = line.replace(shlex.quote(DISPLAY_KV_GID), '"$' + DISPLAY_KV_GID_NAME + '"')
    return setup(c, rank) + line + '\n'


def preflight(c):
    ids = []
    for rank in range(4):
        script = setup(c, rank) + 'test "$(uname -m)" = aarch64\n'
        script += shlex.join(['mkdir', '-p', c['cache_path']]) + '\n'
        script += shlex.join(['test', '-r', checkpoint_path(c, c['model_path']) + '/config.json']) + '\n'
        script += shlex.join(['docker', 'image', 'inspect', '--format', '{{.Os}}/{{.Architecture}}', c['image']]) + " | grep -qx linux/arm64\n"
        if source_pins(c) == LATEST_PINS:
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.performance-bundle"}}', c['image']]) + " | grep -qx ds41-performance-v1\n"
            manifest_sha = hashlib.sha256((HERE/'patches/performance-manifest.json')
                                          .read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            script += shlex.join(docker(c, rank) + [
                'python3', '/opt/ds41/performance-check.py',
                '--expected-manifest-sha256', manifest_sha]) + '\n'
        if any(c.get('roce_optimizations', {}).values()):
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.roce-overlay"}}', c['image']]) + " | grep -qx ds41-roce-v1\n"
        if any(c.get(key, False) for key in DSPARK_PREFILL_OPTIONS):
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.dspark-prefill-overlay"}}', c['image']]) + " | grep -qx ds41-dspark-prefill-v1\n"
        if c.get('b12x_bounded_autotune', False):
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.bounded-autotune"}}', c['image']]) + " | grep -qx ds41-bounded-autotune-v1\n"
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.preparation-memory"}}', c['image']]) + " | grep -qx image-built-v1\n"
        if c.get('prefill_8192_graph', False):
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.prefill-8192-graph"}}', c['image']]) + " | grep -qx ds41-prefill-8192-graph-v2\n"
        if c.get('reduced_tuning', True):
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.b12x-tuning"}}', c['image']]) + " | grep -qx v1\n"
        if c.get('upstream_branch') != 'dev/karmic-kraken':
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.engram-disk"}}', c['image']]) + " | grep -qx v1\n"
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.roce-collective"}}', c['image']]) + " | grep -qx v1\n"
        else:
            for label, commit in (('org.opencontainers.image.revision', c['vllm_commit']),
                                  ('local-inference.b12x.commit', c['b12x_commit'])):
                script += shlex.join(['docker', 'image', 'inspect', '--format',
                                      '{{index .Config.Labels "' + label + '"}}', c['image']])
                script += ' | grep -qx ' + shlex.quote(commit) + '\n'
        script += f"fstype=$(findmnt -n -o FSTYPE -T {shlex.quote(c['model_path'])})\n"
        script += 'case "$fstype" in ext4|xfs|btrfs) ;; *) echo "Model must be local SSD storage, got $fstype"; exit 1;; esac\n'
        for hca in c['hcas']:
            base = f'/sys/class/infiniband/{hca}/ports/1'
            script += shlex.join(['grep', '-q', 'ACTIVE', base + '/state']) + '\n'
            script += shlex.join(['grep', '-q', 'RoCE v2', base + f"/gid_attrs/types/{c['gid_index']}"]) + '\n'
        display = display_kv(c)
        if display:
            label = display['drm_card']
            card = shlex.quote(label)
            # Read-only host state: the launcher never changes modules, boot
            # settings or the firmware reservation, it only refuses to start.
            script += 'if [ -r /sys/module/nvidia_drm/parameters/modeset ]; then\n'
            script += 'test "$(cat /sys/module/nvidia_drm/parameters/modeset)" = Y || ' \
                      '{ echo "nvidia_drm modeset must be Y"; exit 1; }\n'
            script += 'test "$(cat /sys/module/nvidia_drm/parameters/fbdev)" = N || ' \
                      '{ echo "nvidia_drm fbdev must be N"; exit 1; }\n'
            script += 'else echo "WARNING: nvidia_drm parameters unreadable as this user; ' \
                      'verify with sudo and rely on display-check.py"; fi\n'
            script += f'test -c {card} || {{ echo "missing DRM card {label}"; exit 1; }}\n'
            script += shlex.join(['docker', 'image', 'inspect', '--format',
                                  '{{index .Config.Labels "local-inference.display-kv"}}',
                                  c['image']]) + ' | grep -qx v1\n'
            # Fails closed inside the container rather than falling back to
            # ordinary RAM at a utilization the credit already assumes.
            script += shlex.join(docker(c, rank) + ['python3', '/opt/ds41/display-check.py']) + '\n'
        check = '''import ctypes
lib=ctypes.CDLL('liburing.so.2')
ring=ctypes.create_string_buffer(4096)
rc=lib.io_uring_queue_init(8,ring,0)
assert rc==0, f'io_uring denied/unavailable: {rc}'
lib.io_uring_queue_exit(ring)
from b12x.comm import roce
assert roce.is_supported(), 'RoCEnante unsupported'
'''
        script += shlex.join(docker(c, rank) + ['python3', '-c', check]) + '\n'
        if c.get('b12x_bounded_autotune', False):
            counter_check = '''
import torch
from b12x.preparation._memory import _counter, allocated_bytes
assert _counter().__file__.startswith('/opt/ds41/preparation-memory/')
tensor = torch.empty(1024 * 1024, dtype=torch.uint8, device='cuda')
assert allocated_bytes(0) == torch.cuda.memory_allocated(0) >= tensor.numel()
print('Image-built preparation counter matches the live Torch allocator')
'''
            script += shlex.join(['timeout', '90', *docker(c, rank), 'python3', '-c', counter_check]) + '\n'
        script += shlex.join(docker(c, rank) + ['python3', '/opt/ds41/image-check.py', '--gpu']) + '\n'
        script += shlex.join(docker(c, rank) + ['python3', '/opt/ds41/model-check.py', checkpoint_path(c, '/checkpoint')]) + '\n'
        script += shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]) + '\n'
        result = remote(c, rank, script, timeout=600)
        print(f'rank {rank}: {result}', flush=True)
        ids.append(result.splitlines()[-1])
    assert len(set(ids)) == 1, f'Image IDs differ: {ids}'


def fabric(c):
    names = [f'{NAME}-fabric-{rank}' for rank in range(4)]
    launched = []
    try:
        for rank in reversed(range(4)):
            cmd = docker(c, rank, names[rank]) + [
                'torchrun', '--nnodes=4', '--nproc-per-node=1', f'--node-rank={rank}',
                f"--master-addr={c['nodes'][0]['ip']}", f"--master-port={c['master_port'] + 1}",
                '/opt/ds41/roce-check.py']
            remote(c, rank, setup(c, rank) + shlex.join(cmd))
            launched.append(rank)
        def wait(rank):
            return remote(c, rank, shlex.join(['docker', 'wait', names[rank]]), timeout=900)
        with ThreadPoolExecutor(max_workers=4) as pool:
            codes = list(pool.map(wait, range(4)))
        for rank in range(4):
            output = remote(c, rank, shlex.join(['docker', 'logs', names[rank]]) + ' 2>&1')
            print(f'fabric rank {rank}: {output}', flush=True)
            assert codes[rank] == '0' and '"status": "PASS"' in output, 'Fabric qualification failed'
    finally:
        for rank in launched:
            remote(c, rank, shlex.join(['docker', 'rm', '-f', names[rank]]))


def request(c, path, payload=None):
    url = f"http://{c['nodes'][0]['ip']}:{c['port']}{path}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        body = response.read()
        return json.loads(body) if body else None


def smoke(c):
    import math
    def generate(i):
        result = request(c, '/v1/completions', {
            'model': MODEL, 'prompt': f'Question {i}: The four inner planets are',
            'max_tokens': 48, 'temperature': 0, 'logprobs': 1})
        choice = result['choices'][0]
        assert choice['text'].strip(), result
        probs = choice['logprobs']['token_logprobs']
        assert probs and all(p is not None and math.isfinite(p) for p in probs), result
        return result['usage']['completion_tokens']
    start = time.monotonic()
    concurrency = c['max_num_seqs']
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        counts = list(pool.map(generate, range(concurrency)))
    print(json.dumps({'concurrency': concurrency, 'completion_tokens': sum(counts),
                      'wall_seconds': time.monotonic() - start}))
    chat = request(c, '/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': 'Name the four inner planets.'}],
        'max_tokens': 128, 'temperature': 0, 'chat_template_kwargs': {'thinking': False}})
    assert chat['choices'][0]['message'].get('content'), chat
    print('Chat smoke:', chat['choices'][0]['message']['content'])
    for rank in range(4):
        state = remote(c, rank, shlex.join(['docker', 'inspect', '--format', '{{json .State}}', f'{NAME}-{rank}']))
        state = json.loads(state)
        assert state['Running'] and not state['OOMKilled'], (rank, state)
    output = remote(c, 0, shlex.join(['docker', 'logs', f'{NAME}-0']) + ' 2>&1')
    assert 'Using RoCEnante (b12x one-shot RoCE collectives)' in output, 'RoCEnante initialization unconfirmed'
    assert 'RoCEnante all-reduce is live' in output, 'No confirmed RoCEnante dispatch'
    print(f'c{concurrency} smoke and live RoCEnante dispatch passed; inspect generated output for quality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(HERE / 'cluster-karmic-c16.json'))
    parser.add_argument('action', choices=['plan', 'share', 'preflight', 'fabric', 'start', 'smoke', 'status', 'logs', 'stop'])
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    parser.add_argument('--follow', action='store_true', help='Stream new container logs (logs action only)')
    args = parser.parse_args()
    if args.follow and args.action != 'logs':
        parser.error('--follow requires the logs action')
    c = load_config(args.config)
    if args.action == 'plan':
        for rank in range(4):
            print(f'# rank {rank} on {c["nodes"][rank]["host"]}\n{plan(c, rank)}')
    elif args.action == 'share':
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']], text=True).strip()
        for rank in range(4):
            current = remote(c, rank, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]) + ' 2>/dev/null || true')
            if current == image_id:
                print(f'rank {rank}: already has {image_id}')
                continue
            save = subprocess.Popen(['docker', 'save', c['image']], stdout=subprocess.PIPE)
            try:
                subprocess.run(ssh(c, rank) + ['docker', 'load'], stdin=save.stdout, check=True)
            finally:
                save.stdout.close()
            assert save.wait() == 0
            actual = remote(c, rank, shlex.join(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']]))
            assert actual == image_id, (rank, actual, image_id)
            print(f'rank {rank}: {actual}')
    elif args.action == 'preflight':
        preflight(c)
    elif args.action == 'fabric':
        preflight(c)
        fabric(c)
    elif args.action == 'start':
        # Refuse duplicate deployment before probing or allocating model memory.
        for rank in range(4):
            remote(c, rank, f'! docker container inspect {NAME}-{rank} >/dev/null 2>&1')
        preflight(c)
        if c.get('fabric_check', True):
            fabric(c)
        else:
            # Launcher-only skip: serving startup still drives the same
            # coordinated collective path. Qualify with the standalone
            # `fabric` action (best while the fleet is stopped).
            print('Skipping fabric qualification (fabric_check false); '
                  'run `python3 fleet.py --config <config> fabric` to qualify.', flush=True)
        for rank in reversed(range(4)):
            print(remote(c, rank, plan(c, rank)), flush=True)
        deadline = time.monotonic() + c['startup_timeout']
        while True:
            try:
                request(c, '/health')
                break
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Startup timed out; retain containers for logs, then run stop.')
                for rank in range(4):
                    state = remote(c, rank, f"docker inspect --format '{{{{.State.Running}}}}' {NAME}-{rank}")
                    assert state == 'true', f'Rank {rank} exited; run logs --rank {rank}'
                print('Waiting for model startup...', flush=True)
                time.sleep(10)
        smoke(c)
        print(f"Ready: http://{c['nodes'][0]['ip']}:{c['port']}/v1")
    elif args.action == 'smoke':
        smoke(c)
    elif args.action == 'logs':
        command = ['docker', 'logs', '--tail', '200']
        if args.follow:
            command.append('--follow')
        subprocess.run(ssh(c, args.rank) + command + [f'{NAME}-{args.rank}'], check=True)
    else:
        for rank in range(4):
            if args.action == 'stop':
                script = f'if docker container inspect {NAME}-{rank} >/dev/null 2>&1; then docker rm -f {NAME}-{rank}; fi'
            else:
                script = shlex.join(['docker', 'ps', '-a', '--filter', f'name=^/{NAME}-{rank}$'])
            print(f'rank {rank}: {remote(c, rank, script)}')


if __name__ == '__main__':
    main()
