import base64

from io import BytesIO
from pathlib import Path

import numpy as np

from PIL import Image


def encode_image_to_base64(image: np.ndarray | str | Path) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format, a path string,
            or a Path object to an image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, (str, Path)):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
