"""Keep the numbered citations of paper/axis1.md consistent with the verified references.

``.venv/bin/python paper/refs_md.py keys``     body citations [3, 7] -> [@cohan2020, @fister2016] (for editing)
``.venv/bin/python paper/refs_md.py numbers``  [@key] -> [n] with n in order of first citation (IJASCA,
                                               Vancouver style), rewrites the References section from
                                               references_check.json and reference_numbers.json (used by
                                               md2tex.py)

Only keys verified by verify_refs.py can be cited; the reference list is generated from the fetched metadata.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "axis1.md"
NUMBERS = HERE / "reference_numbers.json"
CHECK = HERE / "references_check.json"
HEADING = "## References"


def split_md() -> tuple[str, str]:
    text = MD.read_text()
    i = text.index(HEADING)
    return text[:i], text[i:]


def to_keys() -> None:
    num2key = {v: k for k, v in json.loads(NUMBERS.read_text()).items()}
    body, refs = split_md()
    body = re.sub(r"\[([\d,\s]+)\]", lambda m: "[" + ", ".join(
        "@" + num2key[int(x)] for x in m.group(1).split(",") if x.strip()) + "]", body)
    MD.write_text(body + refs)


def initials(given: str) -> str:
    return "".join(p[0].upper() for p in re.split(r"[\s\-]+", given) if p)


def author(name: str) -> str:
    name = name.replace("\xa0", " ").strip()
    if ", " in name:  # Crossref "Family, Given"
        family, given = name.split(", ", 1)
    else:  # arXiv "Given Family"
        *given_parts, family = name.split()
        given = " ".join(given_parts)
    return f"{family} {initials(given)}".strip()


def entry(m: dict) -> str:
    """Vancouver entry as printed by IJASCA: up to six authors, then the source by publication type."""
    names = [author(a) for a in m["authors"]]
    names = names[:6] + ["et al"] if len(names) > 6 else names
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m["title"])).strip()
    title += "" if title[-1] in ".?!" else "."
    if m["source"] == "arxiv" and "venue" in m.get("overrides", []):  # peer-reviewed version, cited as in TradeGraph
        where = f"In: {m['venue']}; {m['year']}." + (f" p. {m['pages']}." if m.get("pages") else "")
    elif m["source"] == "arxiv":
        where = f"{m['venue']}; {m['year']}. doi: 10.48550/arXiv.{m['arxiv']}"
    elif m.get("type") == "journal-article":
        where = f"{m['venue']}. {m['year']}"
        if m.get("volume"):
            where += f";{m['volume']}" + (f"({m['issue']})" if m.get("issue") else "")
            where += f":{m['pages']}" if m.get("pages") else ""
        where += f". doi: {m['doi']}"
    else:  # proceedings article or book chapter
        where = f"In: {m['venue']}; {m['year']}." + (f" p. {m['pages']}." if m.get("pages") else "")
        where += f" doi: {m['doi']}"
    return f"{', '.join(names)}. {title} {where}"


def to_numbers() -> None:
    checks = json.loads(CHECK.read_text())
    body, _ = split_md()
    order = [k.strip().lstrip("@") for m in re.finditer(r"\[(@[\w]+(?:\s*,\s*@[\w]+)*)\]", body)
             for k in m.group(1).split(",")]
    keys = list(dict.fromkeys(k for k in order if checks.get(k, {}).get("ok")))   # order of first citation
    keys += [k for k, c in checks.items() if c["ok"] and k not in keys]          # uncited: reported below
    key2num = {k: i + 1 for i, k in enumerate(keys)}
    cited: set[str] = set()

    def repl(m: re.Match) -> str:
        ks = [x.strip().lstrip("@") for x in m.group(1).split(",")]
        unknown = [k for k in ks if k not in key2num]
        if unknown:
            sys.exit(f"unverified reference key(s): {unknown}")
        cited.update(ks)
        return "[" + ", ".join(str(n) for n in sorted(key2num[k] for k in ks)) + "]"

    body = re.sub(r"\[(@[\w]+(?:\s*,\s*@[\w]+)*)\]", repl, body)
    if re.search(r"\[@", body):
        sys.exit("malformed citation left in the text")
    uncited = [k for k in keys if k not in cited]
    if uncited:
        sys.exit(f"verified but never cited: {uncited}")
    refs = HEADING + "\n\n" + "\n\n".join(f"[{key2num[k]}] {entry(checks[k])}" for k in keys) + "\n"
    MD.write_text(body + refs)
    NUMBERS.write_text(json.dumps(key2num, indent=1) + "\n")
    print(f"{len(keys)} references, all cited")


if __name__ == "__main__":
    {"keys": to_keys, "numbers": to_numbers}[sys.argv[1]]()
