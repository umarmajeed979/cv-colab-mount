def test_build_architecture_output_shape():
    import torch
    from app.config import settings
    from app.core.model import build_architecture

    model = build_architecture(pretrained_backbone=False)
    model.eval()
    h, w = 64, 64  # small input for a fast test, independent of settings.IMG_SIZE
    dummy = torch.randn(1, 3, h, w)
    with torch.no_grad():
        output = model(dummy)
    logits = output["out"] if isinstance(output, dict) else output
    assert logits.shape == (1, settings.NUM_CLASSES, h, w)


def test_model_loads_returns_none_without_weights(tmp_path, monkeypatch):
    from app.config import settings
    from app.core import model as model_module

    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "does_not_exist.pt"))
    model_module._model = None
    result = model_module.load_model()
    assert result is None
