from __future__ import annotations

import re
import unicodedata
from html import unescape


_TAG_RE = re.compile(r"<[^>]+>")


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unescape(text)
    return " ".join(text.casefold().split()).strip()


def strip_inline_html(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = _TAG_RE.sub("", text)
    return " ".join(text.split()).strip()

