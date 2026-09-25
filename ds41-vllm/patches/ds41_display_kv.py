#!/usr/bin/env python3
"""Display-reserve KV backing for GB10 Sparks; installed into the vLLM package.

The display reserve is firmware memory the OS cannot use, so it is not part of the
profiled ordinary budget. Two halves must agree on one number: credit() admits bytes
to the KV planner, and backing() supplies them from the display span. Both read one
configured value, so a mismatch is impossible by construction.

Disabled by default: DS41_DISPLAY_KV_MIB unset or 0 means credit() returns 0 and
backing() refuses, so a deployment without the host-side state is byte-identical to
today. See DISPLAY-KV-INTEGRATION.md before changing anything here.
"""
import ctypes
import os
from pathlib import Path

QUANTUM = 65536
MAX_DISPLAY_MIB = 1792
_LIBRARY = Path(__file__).resolve().parent / 'libds41_display_kv.so'
_owners = []


def configured_bytes():
    """Display bytes this deployment admits; 0 disables the technique.

    Raises:
        ValueError: If the configured value is outside the allocator's limit.
    """
    mib = int(os.environ.get('DS41_DISPLAY_KV_MIB', '0'))
    if not 0 <= mib <= MAX_DISPLAY_MIB:
        raise ValueError(
            f'DS41_DISPLAY_KV_MIB must be 0..{MAX_DISPLAY_MIB}, got {mib}'
        )
    return mib * 2**20


def span_bytes():
    """The allocator's one fixed span size; the credit never exceeds it.

    The allocator hard-requires exactly this size, so a partial credit still
    creates the full span and only admits less of it to the planner.
    """
    return MAX_DISPLAY_MIB * 2**20


def credit():
    """Bytes to add to the profiled ordinary KV budget; 0 when disabled."""
    return configured_bytes()


class Owner:
    """One contiguous display span, owned for the life of the worker process."""

    def __init__(self, ordinary_bytes):
        import torch

        display = configured_bytes()
        if not display:
            raise RuntimeError('Display KV backing requested while disabled')
        if type(ordinary_bytes) is not int or not 0 <= ordinary_bytes <= 2**30:
            raise ValueError(f'Invalid ordinary prefix: {ordinary_bytes!r}')
        if ordinary_bytes % QUANTUM:
            raise ValueError('Ordinary prefix must be a 64 KiB multiple')
        if not torch.cuda.is_initialized() or torch.cuda.current_device() != 0:
            raise RuntimeError('Initialize the existing CUDA0 context first')
        if _owners:
            raise RuntimeError('Only one display span per worker process is allowed')

        self.lib = ctypes.CDLL(str(_LIBRARY))
        self.lib.ds41_display_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.ds41_display_create.restype = ctypes.c_void_p
        self.lib.ds41_display_pointer.argtypes = [ctypes.c_void_p]
        self.lib.ds41_display_pointer.restype = ctypes.c_uint64
        self.lib.ds41_display_error.restype = ctypes.c_char_p
        self.lib.ds41_display_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.ds41_display_create(ordinary_bytes, span_bytes())
        if not self.handle:
            raise RuntimeError(
                'Display span unavailable: '
                + self.lib.ds41_display_error().decode(errors='replace')
            )
        self.pointer = self.lib.ds41_display_pointer(self.handle)
        self.display_bytes = display
        self.ordinary_bytes = ordinary_bytes
        self.size = ordinary_bytes + span_bytes()

    def __cuda_array_interface__(self):
        return {
            'shape': (self.size,),
            'strides': None,
            'typestr': '|i1',
            'data': (self.pointer, False),
            'version': 3,
        }

    def tensor(self):
        import torch

        tensor = torch.as_tensor(self, device='cuda:0')
        if (
            tensor.data_ptr() != self.pointer
            or tensor.dtype != torch.int8
            or tensor.numel() != self.size
        ):
            raise RuntimeError('CUDA array interface copied or reinterpreted the span')
        return tensor


def backing(size, dtype, device):
    """Back the KV tensors with the display span instead of ordinary RAM.

    Args:
        size: Bytes the KV planner admitted, including any credited display bytes.
        dtype: Requested dtype; only int8 is supported.
        device: Requested device; only cuda:0 is supported.

    Raises:
        RuntimeError: If disabled, or if the admitted size exceeds the span.
    """
    import torch

    if not configured_bytes():
        raise RuntimeError(
            'Display KV backing requested while disabled; refusing to fall back '
            'to ordinary RAM at this utilization'
        )
    if dtype != torch.int8 or torch.device(device) != torch.device('cuda:0'):
        raise RuntimeError(f'Unsupported KV backing request: {dtype} on {device}')
    owner = Owner(ordinary_bytes=0)
    _owners.append(owner)  # process-lifetime ownership, even if a check below fails
    full = owner.tensor()
    if size > owner.size:
        raise RuntimeError(
            f'Admitted KV size {size} exceeds the {owner.size}-byte display span; '
            'lower DS41_DISPLAY_KV_MIB and restart'
        )
    return full[:size]
