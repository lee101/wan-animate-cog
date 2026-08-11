from pathlib import Path

def test_pipeline_uses_no_grad_tensors_for_mutable_diffusers_state():
    source = (Path(__file__).parents[1] / "runtime.py").read_text()
    assert "with torch.no_grad():" in source
    assert "torch.inference_mode()" not in source


def test_fp8_uses_per_tensor_scaling_for_fp32_conditioning_activations():
    source = (Path(__file__).parents[1] / "runtime.py").read_text()
    assert "granularity=PerTensor()" in source
    assert "granularity=PerRow()" not in source


def test_model_cpu_offload_is_enabled_by_default():
    source = (Path(__file__).parents[1] / "runtime.py").read_text()
    assert '_env_bool("WAN_CPU_OFFLOAD", True)' in source
    assert "enable_group_offload(" in source
    assert 'offload_type="block_level"' in source
    assert "num_blocks_per_group=1" in source


def test_preview_area_fits_l40s_activation_budget():
    source = (Path(__file__).parents[1] / "runtime.py").read_text()
    assert '"preview": (512, 384)' in source
