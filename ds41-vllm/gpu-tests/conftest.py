import pytest
import torch

@pytest.fixture(autouse=True)
def require_spark_gpu():
    assert torch.cuda.is_available(), 'Run this qualification on a Spark GPU'
    assert torch.cuda.get_device_capability()[0] == 12, 'SM12x is required'

@pytest.fixture
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config

@pytest.fixture
def workspace_init():
    """Initialize the workspace manager for tests that need it.

    This fixture initializes the workspace manager with the current
    platform's accelerator device if available, and resets it after the test
    completes. Tests that create a full vLLM engine should NOT use this
    fixture as the engine will initialize the workspace manager itself.
    """
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    if torch.accelerator.is_available():
        init_workspace_manager(torch.device(0))
    yield
    reset_workspace_manager()

