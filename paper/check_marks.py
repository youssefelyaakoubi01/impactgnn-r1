"""Check that every modification of the revised article is printed in red.

``.venv/bin/python paper/check_marks.py [paper/axis1.md]``

The journal asks for the changes of a revised manuscript in red. ``md2tex.py`` prints in red the text marked
``{+...+}`` and the blocks between ``<!--rev-->`` and ``<!--/rev-->``, and the references listed in
``new_refs.json``; inside a marked block, ``{=...=}`` keeps in black a passage that is unchanged. This script compares the text that is *not* marked with the submitted version (tag
``v1-submitted``): every unmarked sentence, table row, equation and reference must be found there. Citation numbers
and the numbers of tables, figures, sections, equations and algorithms are ignored in the comparison, because
insertions shift them; these shifts are reported, together with the marked sentences that are in fact unchanged.
Deleted text cannot be coloured and is only counted.

The script also checks that the supplement is entirely inside a marked block and that the colour is present in
``paper.tex`` and ``supplement.tex`` and absent from their ``_clean`` copies. It exits with status 1 when an
unmarked fragment is not in the submitted version.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMITTED = "v1-submitted:paper/axis1.md"
SENT = re.compile(r"(?<!Eq\.)(?<!Eqs\.)(?<=[.;:])\s+(?=[A-Z(])")     # sentence ends, but not after "Eq."


def norm(t: str) -> str:
    """Text without citations, float and section numbers, emphasis marks and repeated spaces."""
    t = re.sub(r"\[(?:@[\w-]+(?:, ?)?)+\]|\[[\d, –-]+\]", "", t)
    t = re.sub(r"(Tables?|Figures?|Algorithm|Sections?) (@[\w-]+|\d+(\.\d+)?)", r"\1 #", t)
    t = re.sub(r"Eqs?\. \(\d+\)", "Eq. (#)", t)
    t = re.sub(r"\\tag\{\d+\}", r"\\tag{#}", t)
    return re.sub(r"\s+", " ", t.replace("**", "").replace("*", "")).strip()


def words(t: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", t))


KEEP = re.compile(r"\{=(.*?)=\}")
CODE = re.compile(r"^(\d+:|Require:)\s*")


def plain(t: str) -> str:
    """Text without the revision marks."""
    return t.replace("{+", "").replace("+}", "").replace("{=", "").replace("=}", "")


def whole(line: str) -> bool:
    """Lines compared as one piece: table rows, equations, references, pseudo-code."""
    return line.startswith(("|", "$$")) or bool(re.match(r"^\[\d+\] |^\d+: |^Require:", line))


def tokens(t: str) -> list[str]:
    return [w.strip(".,;:()") for w in t.split() if w.strip(".,;:()")]


def only_deletions(fragment: str, old_lines: list[list[str]]) -> bool:
    """True when the fragment is a sentence of the submitted text from which words were removed and none added:
    there is then nothing to colour."""
    fw = tokens(fragment)
    for ow in old_lines:
        it = iter(ow)
        if len(fw) <= len(ow) and all(w in it for w in fw):
            return True
    return False


def pieces(text: str) -> list[str]:
    return [text] if whole(text) else SENT.split(text)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "axis1.md"
    new = path.read_text(encoding="utf-8")
    old = subprocess.run(["git", "show", SUBMITTED], cwd=HERE, capture_output=True, text=True, check=True).stdout
    old_n = norm(old)
    old_lines = [tokens(norm(ln)) for ln in old.splitlines() if ln.strip()]
    a, b = new.index("<!--supplement-->"), new.index("<!--/supplement-->")
    article, supp = new[:a] + new[b:], new[a + len("<!--supplement-->"):b]

    # references added in the revision are printed in red by md2tex (new_refs.json); in the Markdown they are the
    # entries of the reference list that the submitted list does not contain
    n_new_refs = len(json.loads((HERE / "new_refs.json").read_text()))
    old_refs = {norm(re.sub(r"^\[\d+\] ", "", ln)) for ln in old.splitlines() if re.match(r"^\[\d+\] ", ln)}

    rev, in_refs = False, False
    bad, cut, same, new_refs, wrong_keep = [], [], [], [], []
    n_all = n_red = n_marked = 0
    for i, line in enumerate(article.split("\n"), 1):
        s = line.strip()
        if s in ("<!--rev-->", "<!--/rev-->"):
            rev = s == "<!--rev-->"
            continue
        if not s or s.startswith(("<!--", "![", "```")):
            continue
        if s == "## References":
            in_refs = True
        n_all += words(plain(line))
        if in_refs and re.match(r"^\[\d+\] ", s):
            if norm(re.sub(r"^\[\d+\] ", "", s)) not in old_refs:
                new_refs.append((i, s))
                n_red += words(s)
            continue
        if rev:                         # the whole line is marked, except the passages kept in black
            for k in KEEP.findall(line):
                if norm(k).strip(" .,;:()") not in old_n:
                    wrong_keep.append((i, k.strip()))
            red = KEEP.sub(" ", line)
            n_red += words(red)
            marked, unmarked = [red], []
        else:
            marked = re.findall(r"\{\+(.*?)\+\}", line)
            unmarked = [p for p in re.split(r"\{\+.*?\+\}", line) if p.strip()]
            n_red += words(" ".join(marked))
        for part in unmarked:
            for c in pieces(part):
                n = norm(c).strip(" .,;:()")
                if len(n) >= 12 and n not in old_n:
                    (cut if only_deletions(n, old_lines) else bad).append((i, c.strip()))
        for part in marked:
            part = CODE.sub("", re.sub(r"^\s*- ", "", part.strip()))      # pseudo-code label, list bullet
            if whole(part) or part.startswith(("#", "**Table", "**Figure", "**Algorithm")) and len(part) < 40:
                continue
            for c in SENT.split(part):
                n = norm(c).strip(" .,;:()")
                if len(n) >= 25:
                    n_marked += 1
                    if n in old_n:
                        same.append((i, c.strip()))

    # a heading that only changed number is not a change of text; report it with the other shifted numbers
    heads_old = {m.group(2): m.group(1) for m in re.finditer(r"^#{2,3} (\d+(?:\.\d+)?) (.+)$", old, flags=re.M)}
    shifted = [(m.group(2), heads_old[m.group(2)], m.group(1))
               for m in re.finditer(r"^#{2,3} (\d+(?:\.\d+)?) (.+)$", article, flags=re.M)
               if m.group(2) in heads_old and heads_old[m.group(2)] != m.group(1)]
    bad = [(i, c) for i, c in bad if not any(c.endswith(f"{new_no} {title}") for title, _, new_no in shifted)]

    new_n = norm(plain(article))
    old_sent = [n for ln in old.splitlines() if ln.strip() and not ln.startswith(("|", "$$", "![", "#", "["))
                for c in SENT.split(ln) for n in [norm(c).strip(" .,;:()")] if len(n) >= 40]
    gone = sum(n not in new_n for n in old_sent)

    problems = len(bad) + len(wrong_keep)
    print(f"article: {n_all} words, {n_red} in red ({100 * n_red / n_all:.0f} %)")
    print(f"unmarked fragments that are not in the submitted version: {len(bad)}")
    for i, c in bad:
        print(f"  line {i}: {c[:160]}")
    print(f"passages kept in black inside a marked block that are not in the submitted version: {len(wrong_keep)}")
    for i, c in wrong_keep:
        print(f"  line {i}: {c[:160]}")
    if len(new_refs) != n_new_refs:
        problems += 1
        print(f"references absent from the submitted list: {len(new_refs)}, but new_refs.json lists {n_new_refs}")
        for i, c in new_refs:
            print(f"  line {i}: {c[:120]}")
    else:
        print(f"new references, printed in red through new_refs.json: {len(new_refs)}")

    body = [ln.strip() for ln in supp.split("\n") if ln.strip()]
    inside = body[:1] == ["<!--rev-->"] and body[-1:] == ["<!--/rev-->"] and body.count("<!--rev-->") == 1
    print("supplement entirely inside a marked block:", "yes" if inside else "NO")
    problems += not inside

    for name, expect in (("paper.tex", True), ("supplement.tex", True), ("paper_clean.tex", False),
                         ("supplement_clean.tex", False)):
        f = HERE / name
        if not f.exists():
            continue
        n = len(re.findall(r"\\textcolor\{red\}|\\color\{red\}", f.read_text()))
        ok = (n > 0) == expect
        print(f"{name}: {n} colour commands", "" if ok else "<- UNEXPECTED")
        problems += not ok

    print("\nfor information")
    print(f"  sentences of the submitted text that are no longer present (deletions are not shown): "
          f"{gone} of {len(old_sent)}")
    print(f"  unmarked sentences that differ only by removed words: {len(cut)}")
    for i, c in cut:
        print(f"    line {i}: {c[:140]}")
    print(f"  marked sentences that are identical to the submitted text: {len(same)} of {n_marked}")
    for i, c in same:
        print(f"    line {i}: {c[:140]}")
    for title, o, n in shifted:
        print(f"  section renumbered, printed in black: {o} -> {n} {title}")
    tags_old, tags_new = re.findall(r"\\tag\{(\d+)\}", old), re.findall(r"\\tag\{(\d+)\}", article)
    print(f"  numbered equations: {len(tags_old)} submitted, {len(tags_new)} now")
    print(f"{problems} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
