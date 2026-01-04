import base64
import json
from typing import Any, Dict, List
from openai import OpenAI

def _b64_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_metadata(
    client: OpenAI,
    image_bytes: bytes,
    color_options: List[str],
    design_options: List[str],
    theme_options: List[str],
) -> Dict[str, Any]:
    """
    GPT Vision extraction:
      returns dict with keys:
        primary_color (str),
        secondary_colors (list[str]),
        design_style (str),
        theme (str),
        suitable_for (str)
    """
    img_b64 = _b64_image(image_bytes)

    # Keep options in prompt, but avoid extremely long content if lists are huge.
    # If your lists are very large (1000s), we can optimize later.
    prompt = f"""
You are extracting wallpaper swatch metadata from an image.

CRITICAL RULES:
- primary_color MUST be exactly one of the COLOR OPTIONS.
- secondary_colors MUST be a JSON array (0 to 6 items), each exactly one of the COLOR OPTIONS.
- design_style MUST be exactly one of the DESIGN STYLE OPTIONS.
- theme MUST be exactly one of the THEME OPTIONS.
- suitable_for can be short free text (e.g., "Living room, Bedroom").
- Do NOT invent new labels outside the options.
- Output ONLY valid JSON (no markdown, no commentary).

Required JSON shape:
{{
  "primary_color": "one color option",
  "secondary_colors": ["color option", "..."],
  "design_style": "one design style option",
  "theme": "one theme option",
  "suitable_for": "free text"
}}

COLOR OPTIONS:
{color_options}

DESIGN STYLE OPTIONS:
{design_options}

THEME OPTIONS:
{theme_options}
""".strip()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            }
        ],
    )

    raw = (resp.output_text or "").strip()
    # Try strict JSON parse. If model returns extra text, attempt to recover.
    return _safe_json_load(raw)

def generate_description(client: OpenAI, accepted_meta: Dict[str, Any]) -> str:
    """
    Description MUST be based only on accepted_meta (no image).
    2-4 sentences, unformatted, focuses on first 4 attributes.
    """
    meta_json = json.dumps(accepted_meta, ensure_ascii=False)

    prompt = f"""
Write a short, unformatted description of this wallpaper swatch.
STRICTLY use ONLY the accepted metadata below (do not add new categories).
Strongly focus on these physical attributes in this order:
1) primary_color
2) secondary_colors
3) design_style
4) theme

Rules:
- 2 to 4 sentences.
- No bullet points, no headings, no markdown.
- Use the exact option strings provided in the metadata.
- Do not invent any new option values.

ACCEPTED METADATA (JSON):
{meta_json}
""".strip()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )
    return (resp.output_text or "").strip()

def validate_categorical(
    meta: Dict[str, Any],
    color_set: set,
    design_set: set,
    theme_set: set,
) -> List[str]:
    """
    Returns list of validation error messages (empty if valid).
    We do NOT block UI if invalid; user can correct via dropdowns.
    """
    errs = []
    pc = meta.get("primary_color")
    sc = meta.get("secondary_colors")
    ds = meta.get("design_style")
    th = meta.get("theme")

    if pc not in color_set:
        errs.append(f"primary_color not in options: {pc}")
    if not isinstance(sc, list):
        errs.append("secondary_colors must be a list")
    else:
        bad = [x for x in sc if x not in color_set]
        if bad:
            errs.append(f"secondary_colors invalid values: {bad}")
    if ds not in design_set:
        errs.append(f"design_style not in options: {ds}")
    if th not in theme_set:
        errs.append(f"theme not in options: {th}")

    return errs

def _safe_json_load(text: str) -> Dict[str, Any]:
    """
    Attempts to parse JSON even if there is stray text.
    Strategy:
      1) direct json.loads
      2) find first '{' and last '}' and parse substring
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    # If still fails, raise a helpful error
    raise ValueError("Could not parse JSON from model output.")
