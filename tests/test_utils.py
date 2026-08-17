def test_preprocess_image_shape():
    import io
    from PIL import Image
    from app.config import settings
    from app.utils.image_utils import preprocess_image

    buf = io.BytesIO()
    Image.new("RGB", (300, 300)).save(buf, format="PNG")
    result = preprocess_image(buf.getvalue())
    h, w = settings.IMG_SIZE
    assert result.shape == (3, h, w)  # CHW


def test_mask_to_color_shape():
    import numpy as np
    from app.utils.image_utils import mask_to_color

    class_map = np.zeros((10, 20), dtype=np.int64)
    color = mask_to_color(class_map)
    assert color.size == (20, 10)  # PIL Image.size is (W, H)


def test_overlay_mask_matches_base_size():
    import numpy as np
    from PIL import Image
    from app.utils.image_utils import overlay_mask

    base = Image.new("RGB", (64, 48))
    class_map = np.zeros((16, 12), dtype=np.int64)
    overlay = overlay_mask(base, class_map)
    assert overlay.size == base.size
