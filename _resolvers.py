"""Register custom OmegaConf resolvers — import before Hydra loads config."""

import re

from omegaconf import OmegaConf


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text[:50].rsplit("-", 1)[0] if len(text) > 50 else text


OmegaConf.register_new_resolver("slugify", _slugify, replace=True)
