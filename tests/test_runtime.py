from pathlib import Path


def test_pipeline_uses_no_grad_tensors_for_mutable_diffusers_state():
    source = (Path(__file__).parents[1] / "runtime.py").read_text()
    assert "with torch.no_grad():" in source
    assert "torch.inference_mode()" not in source
