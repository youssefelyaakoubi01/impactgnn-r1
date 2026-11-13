"""Copy the generated tables (``results/rev1/tables/<name>.md``, written by ``make_tables.py``) into ``axis1.md``.

A table is placed between the marker lines ``<!--table:name-->`` and ``<!--/table:name-->``; everything between
them is replaced. ``.venv/bin/python paper/include_tables.py``
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "results" / "rev1" / "tables"


def main() -> None:
    md = (HERE / "axis1.md").read_text()
    missing = []

    def repl(m: re.Match) -> str:
        path = TABLES / f"{m.group(1)}.md"
        if not path.exists():
            missing.append(m.group(1))
            return m.group(0)
        return f"<!--table:{m.group(1)}-->\n{path.read_text().rstrip()}\n<!--/table:{m.group(1)}-->"

    md, n = re.subn(r"<!--table:([\w-]+)-->\n.*?<!--/table:\1-->", repl, md, flags=re.S)
    (HERE / "axis1.md").write_text(md)
    print(f"{n} table blocks, {len(missing)} without a generated file: {missing}")


if __name__ == "__main__":
    main()
