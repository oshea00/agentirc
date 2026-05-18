"""Text utilities for splitting long responses into chat-safe chunks."""

import textwrap


def split_text(text: str, max_len: int = 400) -> list[str]:
    """Split text into chunks no longer than *max_len* characters.

    Newlines in the source are honoured: each logical paragraph is
    wrapped independently.  Empty paragraphs are dropped so callers
    never receive blank lines as output chunks.

    Returns at least one element — ``["(empty response)"]`` when *text*
    contains only whitespace — so callers can always forward the result
    directly to the transport layer without a length check.

    Args:
        text:    The source string to split.
        max_len: Maximum character length of any returned chunk.

    Returns:
        A non-empty list of strings, each at most *max_len* characters.
    """
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_len:
            lines.append(paragraph)
        else:
            lines.extend(textwrap.wrap(paragraph, max_len))
    return lines or ["(empty response)"]
