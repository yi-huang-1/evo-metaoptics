from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import List, Tuple


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._in_h1 = False
        self._in_h2 = False
        self._in_li = False

        self._title_parts: List[str] = []
        self._h2_parts: List[str] = []

        self._capturing_description = False
        self._description_parts: List[str] = []

        self._capturing_other_names = False
        self._current_li_parts: List[str] = []
        self._other_names: List[str] = []

        self._stop_description = False

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag == "h1":
            self._in_h1 = True
        elif tag == "h2":
            self._in_h2 = True
            self._h2_parts = []
        elif tag == "li":
            self._in_li = True
            self._current_li_parts = []

        if tag in {"p", "br"}:
            if self._capturing_description and not self._stop_description:
                self._description_parts.append("\n")

    def handle_endtag(self, tag: str):  # type: ignore[override]
        if tag == "h1":
            self._in_h1 = False
            self._capturing_description = True
        elif tag == "h2":
            self._in_h2 = False
            heading = _normalize_whitespace("".join(self._h2_parts)).casefold()
            if heading in {"other name", "other names", "external links", "references"}:
                if heading in {"other name", "other names"}:
                    self._capturing_other_names = True
                self._stop_description = True
            else:
                self._capturing_other_names = False
        elif tag == "li":
            self._in_li = False
            if self._capturing_other_names:
                name = _normalize_whitespace("".join(self._current_li_parts))
                if name:
                    self._other_names.append(name)
            self._current_li_parts = []

    def handle_data(self, data: str):  # type: ignore[override]
        if not data:
            return
        text = unescape(data)
        if self._in_h1:
            self._title_parts.append(text)
        elif self._in_h2:
            self._h2_parts.append(text)
        elif self._capturing_description and not self._stop_description:
            self._description_parts.append(text)
        elif self._capturing_other_names and self._in_li:
            self._current_li_parts.append(text)

    def title(self) -> str:
        return _normalize_whitespace("".join(self._title_parts))

    def description(self) -> str:
        raw = "".join(self._description_parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join([line for line in lines if line])

    def other_names(self) -> List[str]:
        return self._other_names


@dataclass(frozen=True)
class MaterialInfo:
    title: str
    description: str
    other_names: Tuple[str, ...]


def extract_material_info(html: str) -> MaterialInfo:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return MaterialInfo(
        title=parser.title(),
        description=parser.description(),
        other_names=tuple(parser.other_names()),
    )
