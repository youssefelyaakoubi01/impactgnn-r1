"""Convert paper/axis1.md (the subset of Markdown it uses) into paper/paper.tex with the layout of an article
published with the IJASCA Author Template 2026 (A4, Times, single column, running header and footer, nomenclature,
numbered equations, Vancouver references cited as (n)), so that both versions always carry the same text and numbers.

Markdown conventions: display equations are one line ``$$ ... \\tag{n} $$``; inline math is ``$...$``; a table
preceded by ``**Nomenclature**`` is the nomenclature box; a fenced block preceded by ``**Algorithm n.** Title`` is
an algorithm whose line numbers are written in the text. The reference list is copied from the Markdown (written by
refs_md.py from the verified metadata), so no BibTeX run is needed.

Revision marks: ``{+text+}`` marks changed or added text inside a line; ``<!--rev-->`` ... ``<!--/rev-->`` (each on
its own line) marks whole blocks (paragraphs, lists, headings, equations, tables, figures, algorithms, references).
``paper.tex`` prints the marked text in red, as requested by the journal for a revised manuscript;
``paper_clean.tex`` is the same document without colour. References whose key is listed in ``new_refs.json`` are
marked as new. Tables, figures and algorithms added in the revision carry a name instead of a number
(``**Table @protocols.**``, cited as ``Table @protocols``); LaTeX numbers all floats in their order of appearance.

Supplement: the part of ``axis1.md`` between ``<!--supplement-->`` and ``<!--/supplement-->`` is written to
``supplement.tex`` (and ``supplement_clean.tex``), with tables numbered S1, S2, ... The article and the supplement
read each other's labels (``xr-hyper``), so ``Table @name`` works across the two files; compile the article, the
supplement, and the article again.

``.venv/bin/python paper/md2tex.py``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = True          # print revision marks in red (False: clean copy)
SUPP = False         # converting the supplement: citations are printed as the numbers of the article's list
REV_ON, REV_OFF = "\x01", "\x02"


def red(tex: str) -> str:
    return r"\textcolor{red}{" + tex + "}" if MARK else tex


def block_colour() -> str:
    return r"\color{red}" if MARK else ""

NUM2KEY = {v: k for k, v in json.loads((HERE / "reference_numbers.json").read_text()).items()}
NEW_REFS = set(json.loads((HERE / "new_refs.json").read_text())) if (HERE / "new_refs.json").exists() else set()
LABEL = r"(\d+|@[\w-]+)"

PREAMBLE = r"""% Generated from axis1.md by md2tex.py. Layout reproduced from an article published with the IJASCA Author
% Template 2026; replace with the official template from the editorial office (ijasca@zuj.edu.jo) if required.
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[a4paper,top=2.5cm,bottom=2.5cm,left=2.2cm,right=2.2cm,headheight=14pt]{geometry}
\usepackage{fancyhdr,titlesec,booktabs,tabularx,graphicx,caption,enumitem,amsmath,xcolor,array}
\IfFileExists{algorithm.sty}{\usepackage{algorithm}}{% same ruled float as algorithm.sty, for installations without it
  \usepackage{float}\floatstyle{ruled}\newfloat{algorithm}{htbp}{loa}\floatname{algorithm}{Algorithm}}
\usepackage{cite}
\usepackage{xr-hyper}
\usepackage[hidelinks]{hyperref}
%%EXTERNAL%%
\renewcommand{\citeleft}{(}\renewcommand{\citeright}{)}
\urlstyle{same}
\graphicspath{{figures/}}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=centering}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parindent}{1.5em}\setlength{\parskip}{3pt}\setlength{\emergencystretch}{1.5em}
\renewcommand{\arraystretch}{1.15}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\titleformat{\section}{\fontsize{14}{17}\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\fontsize{12}{14}\bfseries}{\thesubsection}{0.8em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\definecolor{running}{gray}{0.45}
\pagestyle{fancy}\fancyhf{}
\fancyhead[R]{\footnotesize\itshape\color{running} Int. J. Advance Soft Compu. Appl.}
\fancyfoot[L]{\footnotesize\thepage}
\fancyfoot[R]{\scriptsize\color{running} ISSN 2710-1274 (Print) $\cdot$ 2074-8523 (Online) $\cdot$ Published by
  Al-Zaytoonah University of Jordan (ZUJ)}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0.4pt}
\fancypagestyle{plain}{}
\renewcommand{\refname}{\texorpdfstring{\normalsize REFERENCES}{References}}
\begin{document}
"""

UNICODE = [
    ("ᵃ", r"$^{a}$"), ("ᵇ", r"$^{b}$"), ("±", r"$\pm$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"), ("ρ", r"$\rho$"), ("Σ", r"$\Sigma$"), ("∈", r"$\in$"),
    ("Δ", r"$\Delta$"), ("⁻⁵", r"$^{-5}$"), ("⁻⁴", r"$^{-4}$"), ("⁻³", r"$^{-3}$"), ("⁻²", r"$^{-2}$"), ("²", r"$^{2}$"), ("¹", r"$^{1}$"), ("→", r"$\rightarrow$"), ("−", r"$-$"),
    ("—", "---"), ("–", "--"), ("…", r"\ldots{}"), ("“", "``"), ("”", "''"), ("’", "'"), ("·", r"$\cdot$"),
]


def esc(text: str) -> str:
    """Escape LaTeX specials outside of already-converted commands."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")):
        text = text.replace(a, b)
    return text


def inline(text: str) -> str:
    """Markdown inline markup -> LaTeX (math, code, bold, italic, citations, unicode)."""
    holders: list[str] = []

    def hold(s: str) -> str:
        holders.append(s)
        return f"\x00{len(holders) - 1}\x00"

    text = text.replace("{+", REV_ON).replace("+}", REV_OFF)
    text = re.sub(r"\$([^$]+)\$", lambda m: hold("$" + m.group(1) + "$"), text)
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + esc(m.group(1)) + "}"), text)
    text = re.sub(r"\bT0\b", lambda m: hold("$T_0$"), text)                      # labels of the generated tables
    text = re.sub(r"\*k\* = (\d)", lambda m: hold(r"\mbox{\emph{k} = " + m.group(1) + "}"), text)
    if SUPP:        # the supplement has no reference list: the numbers are those of the article
        text = re.sub(r"\[([\d,\s]+)\]", lambda m: hold("~(" + ", ".join(
            x.strip() for x in m.group(1).split(",") if x.strip()) + ")"), text)
    else:
        text = re.sub(r"\[([\d,\s]+)\]", lambda m: hold(r"~\cite{" + ",".join(
            NUM2KEY[int(x)] for x in m.group(1).split(",") if x.strip()) + "}"), text)
    text = re.sub(r"https?://\S+?(?=[.,;:)]*(?:\s|$))", lambda m: hold(r"\url{" + m.group(0) + "}"), text)
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", text)
    text = re.sub(r"(?<=[A-Za-z])–(?=[A-Za-z])", r"--\\allowbreak{}", text)  # paper–author–paper may break
    text = re.sub(r'"([^"]+)"', r"``\1''", text)
    for a, b in UNICODE:
        text = text.replace(a, b)
    text = re.sub(r"(\d) \\%", r"\1\\,\\%", text)
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1{,}\2", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: holders[int(m.group(1))], text)
    text = text.replace(REV_ON, r"\textcolor{red}{" if MARK else "").replace(REV_OFF, "}" if MARK else "")
    return text.replace(" ~\\cite", "~\\cite")


def cells_of(rows: list[str]) -> list[list[str]]:
    """Pipe-table rows -> cells, without the separator row."""
    return [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s\-|:]+\|$", r)]


def table(rows: list[str], caption: str, label: str, rev: bool = False, note: str = "") -> str:
    """Pipe table -> tabularx over the full text width. Widths are computed in units of one digit: every column
    first receives what it cannot break (numbers, "mean ± std", intervals, the longest word), and the remaining
    width goes to the columns whose text wraps. The font size is the largest one at which the unbreakable parts fit.
    The first column is left-aligned, the others centred. ``note`` is printed under the table, inside the float."""
    cells = cells_of(rows)
    head, body = cells[0], cells[1:]
    ncol = len(head)
    bare = lambda c: re.sub(r" (\[[\d, ]+\])", lambda m: "_" + m.group(1).replace(" ", ""),  # noqa: E731
                            c.replace("{+", "").replace("+}", "").replace("*", "").strip())  # a citation stays with its word
    numeric = re.compile(r"^[−+-]?\d[\d.,]*( (±|to|of) [\d.]+)?( ?%)?$")
    interval = re.compile(r"^([−+-]?\d[\d.]* )?\[[^\]]+\][ᵃᵇ]?$")
    wlen = lambda w: sum(1.38 if ch.isupper() else 1.0 if ch.isdigit() else 0.93 if ch.isalpha() else 0.7  # noqa: E731
                         for ch in w)                             # widths of the characters, in digits
    text_len = lambda c: (0.87 * len(c) if numeric.match(c) or interval.match(c)  # noqa: E731
                          else sum(wlen(w) for w in c.split()) + 0.5 * c.count(" "))
    word = lambda c: max((wlen(w) for w in c.split()), default=0)  # noqa: E731
    cols = [[bare(r[j]) for r in body if j < len(r)] for j in range(ncol)]
    bold = lambda w: sum(1.6 if ch.isupper() else 1.0 if ch.isdigit() else 1.08 if ch.isalpha() else 0.75  # noqa: E731
                         for ch in w)                             # headers are bold; capitals are wide
    head_word = [max((bold(w) for w in bare(h).split()), default=0) + 1 for h in head]
    head_len = [sum(bold(w) for w in bare(h).split()) + 0.5 * bare(h).count(" ") for h in head]
    nat = [max([text_len(c) + 1 for c in col] + [head_word[j], head_len[j] / 2 + 1])   # a header on two lines at most
           for j, col in enumerate(cols)]

    def minimum(one_line: bool) -> list[float]:
        out = []
        for j, col in enumerate(cols):
            fixed = [(0.87 * len(c) if numeric.match(c) or (one_line and interval.match(c)) else word(c)) + 1
                     for c in col]                                # signs, points and brackets are narrower than digits
            out.append(max(fixed + [head_word[j], 3]))
        return out

    sep = 3 if ncol >= 7 else 4 if ncol >= 6 else 6               # \tabcolsep in points
    sizes = ((r"\footnotesize", 4.5), (r"\scriptsize", 4.0))      # width of a digit at 9 pt and at 8 pt
    if len(body) > 20 and any(len(c) > 30 for col in cols for c in col):   # long tables with wrapping text: small font
        sizes = sizes[1:]

    def wrapped(c: str, width: float, scale: float = 1.0) -> int:
        """Number of lines of a cell in a column of that width (greedy word wrapping)."""
        if numeric.match(c) or interval.match(c):
            return 1
        n, used = 1, 0.0
        for w in c.split():
            need = scale * wlen(w)
            if used and used + 0.5 + need > width - 1:
                n, used = n + 1, need
            else:
                used += (0.5 if used else 0) + need
        return n

    def tall(j: int, width: float) -> int:
        return max([wrapped(c, width) for c in cols[j]] + [wrapped(bare(head[j]), width, 1.2)])

    def layout(mins: list[float], budget: float) -> tuple[list[float], float]:
        """Free width goes, digit by digit, to the column whose longest cell wraps over the most lines."""
        width = list(mins)
        free = budget - sum(width)
        while free >= 1:
            lines = [tall(j, width[j]) if nat[j] - width[j] > 0.5 else 0 for j in range(ncol)]
            j = max(range(ncol), key=lambda q: (lines[q], nat[q] - width[q]))
            if lines[j] <= 1:
                break
            width[j] += 1
            free -= 1
        return width, max(tall(j, width[j]) for j in range(ncol))

    options = []
    for one_line in (True, False):                                # intervals on one line if a font size allows it
        for n, (size, digit) in enumerate(sizes):
            mins = minimum(one_line)
            budget = 468 / digit - ncol * 2 * sep / digit
            if sum(mins) <= budget:
                width, lines = layout(mins, budget)
                options.append(((not one_line, max(lines, 2), n), size, one_line, width))
    if not options:
        mins, budget = minimum(False), 468 / 4.0 - ncol * 2 * sep / 4.0
        options.append(((0,), r"\scriptsize", False, layout(mins, max(budget, sum(mins)))[0]))
    _, size, one_line, width = min(options, key=lambda o: o[0])   # then the largest font without tall cells
    total = sum(width)
    align = lambda j: r"\raggedright" if j == 0 else r"\centering"  # noqa: E731
    spec = "".join(">{" + align(j) + r"\arraybackslash\hsize=" + f"{ncol * w / total:.3f}" + r"\hsize}X"
                   for j, w in enumerate(width))
    out = [r"\begin{table}[htbp]", r"\centering" + (block_colour() if rev else ""), size,
           rf"\setlength{{\tabcolsep}}{{{sep}pt}}",
           rf"\caption{{{red(inline(caption)) if rev else inline(caption)}}}\label{{{label}}}",
           rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}", r"\toprule",
           " & ".join(r"\textbf{" + inline(h) + "}" for h in head) + r" \\", r"\midrule"]

    def cell(c: str, j: int) -> str:
        b = bare(c)
        keep = (j > 0 and numeric.match(b)) or (one_line and interval.match(b))
        return r"\mbox{" + inline(c) + "}" if keep else inline(c)

    out += [" & ".join(cell(c, j) for j, c in enumerate(r)) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabularx}"]
    if note:
        out.append(r"\par\smallskip\raggedright\scriptsize " + (red(inline(note)) if rev else inline(note)))
    out.append(r"\end{table}")
    return "\n".join(out)


def nomenclature(rows: list[str]) -> str:
    """Four-column symbol table under a NOMENCLATURE title, as on the first page of IJASCA articles; the
    Markdown header row (Symbol | Description) is not printed."""
    sym = r">{\raggedright\arraybackslash}p{0.12\linewidth}"
    out = [r"\par\medskip\noindent{\small", rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}{sym}Y{sym}Y@{{}}}}", r"\toprule",
           r"\multicolumn{4}{@{}l}{\textbf{NOMENCLATURE}} \\", r"\midrule"]
    out += [" & ".join(inline(c) for c in r) + r" \\" for r in cells_of(rows)[1:]]
    out += [r"\bottomrule", r"\end{tabularx}\par}", r"\medskip"]
    return "\n".join(out)


def algorithm(body: list[str], caption: str, num: str, rev: bool = False) -> str:
    """Fenced pseudo-code -> algorithm float. Lines are ``n: text`` (or ``Require: text``); the nesting level is
    read from the column where the text starts, in steps of four characters."""
    out = [r"\begin{algorithm}[htbp]" + (block_colour() if rev else ""),
           rf"\caption{{{red(inline(caption.rstrip('.'))) if rev else inline(caption.rstrip('.'))}}}\label{{alg:{num}}}",
           r"\small"]
    for ln in body:
        label, ws, text = re.match(r"^(\d+:|Require:)(\s*)(.*)$", ln).groups()
        if label == "Require:":
            out.append(r"\noindent\hangindent=4.8em\hangafter=1\textbf{Require:} " + inline(text) + r"\par")
            continue
        level = max(0, (len(label) + len(ws) - 3) // 4)
        out.append(rf"\noindent\hangindent={2.2 + 1.5 * level}em\hangafter=1" +
                   rf"\makebox[1.6em][r]{{{label}}}\hspace{{0.6em}}\hspace*{{{1.5 * level}em}}" +
                   inline(text) + r"\par")
    out.append(r"\end{algorithm}")
    return "\n".join(out)


def reference(ln: str) -> str:
    """``[n] text. doi: X`` -> bibitem with a DOI link."""
    new = "{+" in ln or NUM2KEY[int(re.match(r"^\[(\d+)\]", ln.replace("{+", "")).group(1))] in NEW_REFS
    num, text = re.match(r"^\[(\d+)\] (.*)$", ln.replace("{+", "").replace("+}", "")).groups()
    m = re.match(r"^(.*) doi: (\S+)$", text)
    doi = rf" doi: \href{{https://doi.org/{m.group(2)}}}{{\nolinkurl{{{m.group(2)}}}}}" if m else ""
    body = inline(m.group(1) if m else text) + doi
    return rf"\bibitem{{{NUM2KEY[int(num)]}}} " + (red(body) if new else body)


SUPP_INTRO = ("This file accompanies the article of the same title. It contains the detailed results of the "
              "secondary analyses and the additional tables. Its tables are numbered with the prefix S. Section, "
              "table, figure and equation numbers without this prefix refer to the article.")


def split_supplement(md: str) -> tuple[str, str]:
    """(article, supplement) sources. The supplement starts with the title and the authors of the article."""
    a, b = md.index("<!--supplement-->"), md.index("<!--/supplement-->")
    body = md[a + len("<!--supplement-->"):b]
    head = md.splitlines()
    title = next(ln for ln in head if ln.startswith("# "))
    authors = next(ln for ln in head if ln.startswith("**Youssef"))
    supp = (title.replace("# ", "# Supplementary Material. ", 1) + "\n\n" + authors + "\n\n{+" + SUPP_INTRO + "+}\n"
            + body)
    return md[:a] + md[b + len("<!--/supplement-->"):], supp


def convert(md: str, external: str = "", supplement: bool = False) -> str:
    global SUPP
    SUPP = supplement
    lines = md.splitlines()
    pre = PREAMBLE.replace("%%EXTERNAL%%", rf"\externaldocument{{{external}}}" if external else "")
    if supplement:
        pre = pre.replace(r"\begin{document}", r"\renewcommand{\thetable}{S\arabic{table}}"
                          r"\renewcommand{\thefigure}{S\arabic{figure}}" + "\n" + r"\begin{document}")
    out = [pre]
    i = 0
    pending_caption = None
    in_list = None
    in_decl = False
    in_refs = False
    noindent_next = False
    rev = False          # inside a <!--rev--> ... <!--/rev--> block
    R = lambda tex: red(tex) if rev else tex  # noqa: E731

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(rf"\end{{{in_list}}}")
            in_list = None

    while i < len(lines):
        ln = lines[i]
        if ln.strip() in ("<!--rev-->", "<!--/rev-->"):
            rev = ln.strip() == "<!--rev-->"
            i += 1
            continue
        if ln.startswith("<!--"):
            i += 1
            continue
        if ln.startswith("# "):
            out.append(r"{\raggedright\fontsize{18}{22}\selectfont\bfseries " + inline(ln[2:]) + r"\par}" + "\n"
                       + r"\vspace{10pt}")
            i += 1
            continue
        if ln.startswith("**Youssef"):
            authors = re.sub(r"(\S)¹(\\\*)?", lambda m: m.group(1) + r"$^{1" + (",*" if m.group(2) else "") + "}$",
                             ln.strip("*"))
            out.append(r"{\raggedright\fontsize{12}{15}\selectfont " + authors + r"\par}" + "\n" + r"\vspace{6pt}")
            i += 1
            continue
        if ln.startswith("¹ "):
            corr = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].startswith("\\*") else ""
            out.append(r"{\raggedright\footnotesize\itshape $^{1}$ " + inline(ln[2:]) +
                       (r"\\ $^{*}$ " + inline(corr.lstrip("\\* ").strip()) if corr else "") + r"\par}" + "\n"
                       + r"\vspace{10pt}")
            i += 2 if corr else 1
            continue
        if ln == "## Abstract":
            close_list()
            out.append(r"\noindent\textbf{ABSTRACT}\par")
            noindent_next = True
            i += 1
            continue
        if ln.startswith("**Keywords:**"):
            out.append(r"\noindent\textbf{Keywords:} " + inline(ln.replace("**Keywords:**", "").strip()) + r"\par")
            i += 1
            continue
        if ln == "**Nomenclature**":
            pending_caption = ("Nomenclature", "", "")
            i += 1
            continue
        m = re.match(r"^(#{2,3}) (\d+(?:\.\d+)?\s+)?(.*)$", ln)
        if m:
            close_list()
            title = m.group(3)
            if title == "References":
                if in_decl:
                    out.append(r"\endgroup")
                    in_decl = False
                in_refs = True
                out.append(r"\clearpage" + "\n" + r"{\small\begin{thebibliography}{99}\setlength{\itemsep}{1pt}")
            elif title == "Declarations":
                out.append(r"\clearpage" + "\n" + r"\section*{\texorpdfstring{\normalsize DECLARATIONS}{Declarations}}" + "\n"
                           + r"\begingroup\small")
                in_decl = True
            elif m.group(1) == "##":
                out.append((r"\clearpage" + "\n" if title.startswith("Appendix") else "") +   # floats of the text first
                           (r"\section*{" if not m.group(2) else r"\section{") + R(inline(title)) + "}")
            else:
                out.append(r"\subsection{" + R(inline(title)) + "}")
            i += 1
            continue
        if in_refs:
            if ln.strip():
                out.append(reference(("{+" + ln + "+}") if rev and "{+" not in ln else ln))
            i += 1
            continue
        mcap = re.match(r"^\*\*(Table|Figure|Algorithm) " + LABEL + r"\.\*\* (.*)$", ln)
        if mcap:
            # table and algorithm captions precede their block; figure captions follow the image (handled below)
            pending_caption = (mcap.group(1), mcap.group(2).lstrip("@"), mcap.group(3))
            i += 1
            continue
        meq = re.match(r"^\$\$(.*?)\\tag\{(\d+)\}\s*\$\$$", ln.strip())
        if meq:
            while out and not out[-1].strip():  # keep the equation inside the paragraph that introduces it
                out.pop()
            eq = (r"\begin{equation*}" + meq.group(1).strip() + rf"\tag{{{meq.group(2)}}}\label{{eq:{meq.group(2)}}}"
                  + r"\end{equation*}")
            out.append(("{" + block_colour() + eq + "}") if rev else eq)
            noindent_next = True
            i += 1
            continue
        if ln.startswith("```"):
            body = []
            i += 1
            while not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            if not (pending_caption and pending_caption[0] == "Algorithm"):
                raise SystemExit(f"line {i}: fenced block without an **Algorithm n.** caption")
            out.append(algorithm(body, pending_caption[2], pending_caption[1], rev))
            pending_caption = None
            i += 1
            continue
        if ln.startswith("!["):
            path = re.search(r"\((.*?)\)", ln).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = re.match(r"^\*\*Figure " + LABEL + r"\.\*\* (.*)$", lines[j])
            out.append(r"\begin{figure}[htbp]\centering" + (block_colour() if rev else "") +
                       r"\includegraphics[width=\linewidth]{" +
                       Path(path).stem + "}" + rf"\caption{{{R(inline(cap.group(2)))}}}" +
                       rf"\label{{fig:{cap.group(1).lstrip('@')}}}" + r"\end{figure}")
            i = j + 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            if pending_caption and pending_caption[0] == "Nomenclature":
                out.append(nomenclature(rows))
            elif pending_caption and pending_caption[0] == "Table":
                note = ""        # a paragraph that starts with the first note mark belongs to the table
                if i + 1 < len(lines) and not lines[i].strip() and lines[i + 1].startswith(("ᵃ ", "ᵇ ", "\"Aggregates\"", "Abbreviations:",
                                                                                     "Registered contrasts", "None of")):
                    note = lines[i + 1]
                    i += 2
                out.append(table(rows, pending_caption[2], f"tab:{pending_caption[1]}", rev, note))
            else:
                raise SystemExit(f"line {i}: table without a **Table n.** caption")
            pending_caption = None
            continue
        mli = re.match(r"^(\s*)(-|\d+\.) (.*)$", ln)
        if mli:
            kind = "itemize" if mli.group(2) == "-" else "enumerate"
            if in_list != kind:
                close_list()
                # author contributions: indented names without bullets, as in IJASCA declarations
                out.append(rf"\begin{{{kind}}}" + ("[label={},leftmargin=1.5em,itemsep=1pt]" if in_decl else ""))
                in_list = kind
            out.append(r"\item " + R(inline(mli.group(3))))
            i += 1
            continue
        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue
        close_list()
        out.append((r"\noindent " if noindent_next or in_decl else "") + R(inline(ln)))
        noindent_next = False
        i += 1
    close_list()
    if in_refs:
        out.append(r"\end{thebibliography}}")
    out.append(r"\end{document}")
    tex = "\n".join(out)
    # references to tables, figures, algorithms, equations and sections in the text
    name = r"(\d+|@(?:[\w-]|\\_)+)"                       # a number, or @label (underscores arrive escaped)
    lab = lambda m, i=1: m.group(i).lstrip("@").replace("\\_", "_")  # noqa: E731
    tex = re.sub("Tables " + name + " and " + name,
                 lambda m: r"Tables~\ref{tab:" + lab(m, 1) + r"} and~\ref{tab:" + lab(m, 2) + "}", tex)
    tex = re.sub("Table " + name, lambda m: r"Table~\ref{tab:" + lab(m) + "}", tex)
    tex = re.sub("Figure " + name, lambda m: r"Figure~\ref{fig:" + lab(m) + "}", tex)
    tex = re.sub("Algorithm " + name, lambda m: r"Algorithm~\ref{alg:" + lab(m) + "}", tex)
    tex = re.sub(r"(Eqs?\.) \(", r"\1~(", tex)
    tex = re.sub(r"Section (\d+(?:\.\d+)?)", r"Section~\1", tex)
    return tex


def main() -> None:
    global MARK
    article, supp = split_supplement((HERE / "axis1.md").read_text())
    if re.search(r"\[@", supp):
        raise SystemExit("run refs_md.py numbers first: the supplement prints reference numbers")
    for suffix, mark in (("", True), ("_clean", False)):
        MARK = mark
        (HERE / f"paper{suffix}.tex").write_text(convert(article, external=f"supplement{suffix}"))
        (HERE / f"supplement{suffix}.tex").write_text(convert(supp, external=f"paper{suffix}", supplement=True))
        print(HERE / f"paper{suffix}.tex", HERE / f"supplement{suffix}.tex")


if __name__ == "__main__":
    main()
