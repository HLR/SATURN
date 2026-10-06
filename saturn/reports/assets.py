"""Static assets for the HTML debug reports.

The CSS, JS and HTML skeletons live as plain files under ``templates/`` so
that ``debug_report.py`` / ``sections.py`` contain only Python. Blocks inside
a template file are delimited by ``@@block <name>`` marker lines (``//`` in
JS, ``<!-- -->`` in HTML); a block's text runs from the line after its marker
up to the next marker, minus one trailing newline. Placeholders use
:class:`string.Template` syntax (``$name``) — no third-party dependency.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from string import Template
from typing import Dict

_BLOCK_RE = re.compile(r"^\s*(?://|<!--|/\*)\s*@@block\s+(\w+)")


def template_text(name: str) -> str:
    """Return the raw text of ``saturn/reports/templates/<name>``."""
    return (
        resources.files("saturn.reports")
        .joinpath(f"templates/{name}")
        .read_text(encoding="utf-8")
    )


def _parse_blocks(text: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    cur = None
    buf = []

    def _flush():
        if cur is not None:
            body = "".join(buf)
            if body.endswith("\n"):
                body = body[:-1]
            blocks[cur] = body

    for line in text.splitlines(keepends=True):
        m = _BLOCK_RE.match(line)
        if m:
            _flush()
            cur = m.group(1)
            buf = []
        elif cur is not None:
            buf.append(line)
    _flush()
    return blocks


@lru_cache(maxsize=None)
def blocks(name: str) -> Dict[str, str]:
    """Parsed ``@@block`` sections of a template file."""
    return _parse_blocks(template_text(name))


def template_block(name: str, key: str) -> Template:
    """A :class:`string.Template` for block ``key`` of template file ``name``."""
    return Template(blocks(name)[key])


def script_tag(js: str) -> str:
    """Wrap an inline JS snippet in a ``<script>`` element."""
    return f"<script>\n{js}\n</script>"


CSS = template_text("report.css")
INDEX_CSS = CSS + template_text("index.css")
