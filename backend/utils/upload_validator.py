"""
backend/utils/upload_validator.py

Common upload validation logic for PicWise image endpoints (STEP 7 Part C).
Ensures consistent validation across /api/food/analyze, /api/personal-care/analyze,
and /api/analyze:
- Category validation
- Image presence and non-empty filename
- Allowed extensions and MIME types
- Non-empty payload
- Maximum file size (16MB ceiling -> 413)
- Pillow image integrity verification
- Pixel safety check (MAX_IMAGE_PIXELS -> 400)
"""

import io
from typing import Any, Dict, List, Optional, Sequence, Tuple
from PIL import Image

from backend import config


def is_allowed_image(mimetype: str, filename: str) -> bool:
    """Verifies that the uploaded file has a supported extension and MIME type."""
    if not filename or not mimetype:
        return False
    lower_name = filename.lower()
    has_allowed_ext = any(lower_name.endswith(ext) for ext in config.ALLOWED_IMAGE_EXTENSIONS)
    has_allowed_mime = mimetype in config.ALLOWED_IMAGE_MIMETYPES
    return has_allowed_ext and has_allowed_mime


def validate_image_upload(
    request: Any,
    expected_category: Optional[str] = None,
    allowed_categories: Optional[Sequence[str]] = None,
    max_bytes: Optional[int] = None,
    include_success_flag: bool = True,
) -> Tuple[Optional[bytes], Optional[str], Optional[Tuple[Dict[str, Any], int]]]:
    """
    Validates an incoming multipart/form-data upload request.

    Parameters:
        request: Flask request object.
        expected_category: Single required category (e.g. 'food' or 'personal_care').
        allowed_categories: Allowed category choices (e.g. ('food', 'personal_care')).
        max_bytes: Maximum allowed byte size (defaults to config.MAX_CONTENT_LENGTH).
        include_success_flag: If True, error dict includes {"success": False}.

    Returns:
        (image_bytes, validated_category, None) on success, or
        (None, None, (error_dict, status_code)) on failure.
    """
    max_limit = max_bytes or config.MAX_CONTENT_LENGTH

    def make_error(msg: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
        # Note: 413 error responses always include "success": False
        if include_success_flag or status_code == 413:
            return {"error": msg, "success": False}, status_code
        return {"error": msg}, status_code

    # 1. Category validation
    category = request.form.get("category")
    if not category or not category.strip():
        return None, None, make_error("Product category is required in form field 'category'.", 400)

    category = category.strip().lower()

    if expected_category is not None:
        if category != expected_category.lower():
            return (
                None,
                None,
                make_error(
                    f"Invalid category '{category}'. This endpoint strictly handles '{expected_category}' analysis.",
                    400,
                ),
            )
    elif allowed_categories is not None:
        allowed_lower = [c.lower() for c in allowed_categories]
        if category not in allowed_lower:
            return (
                None,
                None,
                make_error(
                    f"Invalid category '{category}'. Allowed values are 'food' or 'personal_care'.",
                    400,
                ),
            )

    # 2. Image field exists & 3. Filename is non-empty
    image = request.files.get("image")
    if image is None or image.filename == "":
        return None, None, make_error("Image file is required in form field 'image'.", 400)

    # 4. Extension & 5. MIME type is supported
    if not is_allowed_image(image.mimetype, image.filename):
        return None, None, make_error("Only JPG, JPEG, PNG, and WEBP images are supported.", 400)

    # 6. Image bytes are non-empty
    image_bytes = image.read()
    if not image_bytes or len(image_bytes) == 0:
        return None, None, make_error("Uploaded image file is empty.", 400)

    # 7. File size is within MAX_CONTENT_LENGTH (16MB -> 413)
    if len(image_bytes) > max_limit:
        return (
            None,
            None,
            make_error("Uploaded image file exceeds the maximum allowed size of 16MB.", 413),
        )

    # 8. Image content is a valid image using Pillow & 9. Pixel safety check
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        if width * height > config.MAX_IMAGE_PIXELS:
            return None, None, make_error("Image pixel count exceeds maximum allowed limit.", 400)
        pil_img.verify()
    except Image.DecompressionBombError:
        return None, None, make_error("Image pixel count exceeds maximum allowed limit.", 400)
    except Exception:
        return None, None, make_error("Invalid or corrupt image file.", 400)

    return image_bytes, category, None
