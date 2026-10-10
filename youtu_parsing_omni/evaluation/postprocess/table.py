"""Table conversion for document Markdown: OTSL / Markdown -> HTML, HTML validation."""

import re
from html import escape as _html_escape

import lxml.html
import markdown


# ---------------------------------------------------------------------------
# OTSL struct → HTML  (ported from convert.py, identical algorithm)
# ---------------------------------------------------------------------------

# 行内 token 形如 <start>0<content>xxx  | <left>1<content>  | <up>1<content>  | <left_up>1<content>
_TOKEN_RE = re.compile(
    r'<(start|left|up|left_up)>\s*(\d+)\s*<content>'
)


def _parse_struct_to_grid(otsl_text):
    """
    将 <start>/<left>/<up>/<left_up>+<content> 结构按行解析为 grid。
    返回:
        grid: list[list[dict]]   每个单元格 {"role": "start|left|up|left_up", "col": int, "text": str}
              其中 text 仅在 role==start 有意义；其它格子是合并的占位。
        total_cols: int
    若解析失败返回 (None, 0)。
    """
    lines = [ln for ln in otsl_text.split('\n') if ln.strip()]
    grid = []
    for ln in lines:
        # 找到所有 token 起始位置
        matches = list(_TOKEN_RE.finditer(ln))
        if not matches:
            continue
        row = []
        for i, m in enumerate(matches):
            role = m.group(1)
            try:
                col = int(m.group(2))
            except ValueError:
                col = len(row)
            start_text = m.end()
            end_text = matches[i + 1].start() if i + 1 < len(matches) else len(ln)
            cell_text = ln[start_text:end_text]
            row.append({"role": role, "col": col, "text": cell_text})
        grid.append(row)

    if not grid:
        return None, 0

    # 计算总列数：取每行单元格数最大值，或单元格最大 col+1
    total_cols = max(
        max((c["col"] for c in row), default=-1) + 1 if row else 0
        for row in grid
    )
    total_cols = max(total_cols, max(len(r) for r in grid))
    return grid, total_cols


def _grid_to_html(grid, total_cols):
    """
    将解析后的 grid 转回 HTML 表格，处理 colspan / rowspan。
    """
    total_rows = len(grid)
    if total_rows == 0 or total_cols == 0:
        return ""

    # 构造规整矩阵 cells[r][c] = role/text 信息（按真实列对齐）
    matrix = [[None] * total_cols for _ in range(total_rows)]
    for r, row in enumerate(grid):
        # 行内 token 是按列顺序排列，col 字段是声明列号；这里按声明列号填，缺失视作 None
        # 如果 col 越界或重复，就退化为按序填入下一个空槽
        used_cols = set()
        for cell in row:
            c = cell["col"]
            if c < 0 or c >= total_cols or c in used_cols or matrix[r][c] is not None:
                # 找第一个空槽
                for cc in range(total_cols):
                    if matrix[r][cc] is None and cc not in used_cols:
                        c = cc
                        break
                else:
                    continue
            matrix[r][c] = cell
            used_cols.add(c)

    # 计算 colspan/rowspan
    visited = [[False] * total_cols for _ in range(total_rows)]
    html_rows = []
    for r in range(total_rows):
        row_html = ["<tr>"]
        for c in range(total_cols):
            if visited[r][c]:
                continue
            cell = matrix[r][c]
            if cell is None:
                # 无内容：作为空 td 占位
                row_html.append("<td></td>")
                visited[r][c] = True
                continue
            role = cell["role"]
            if role != "start":
                # 不应作为 td 起点；但前面 visited 没标，说明被合并的左/上单元缺失，作为空 td 输出
                row_html.append("<td></td>")
                visited[r][c] = True
                continue

            # 计算 colspan
            colspan = 1
            while c + colspan < total_cols:
                nxt = matrix[r][c + colspan]
                if nxt is not None and nxt["role"] in ("left", "left_up"):
                    colspan += 1
                else:
                    break
            # 计算 rowspan
            rowspan = 1
            while r + rowspan < total_rows:
                nxt = matrix[r + rowspan][c]
                if nxt is not None and nxt["role"] in ("up", "left_up"):
                    rowspan += 1
                else:
                    break

            # 标记 visited
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    if rr < total_rows and cc < total_cols:
                        visited[rr][cc] = True

            text = cell["text"]
            esc = text.replace("<br>", " ").strip()
            attrs = ""
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            row_html.append(f"<td{attrs}>{esc}</td>")
        row_html.append("</tr>")
        html_rows.append("".join(row_html))

    return "<table>" + "".join(html_rows) + "</table>"


def otsl_to_html(otsl_str: str) -> str:
    """Convert OTSL struct format to HTML table.

    Ported from convert.py ``otssl_struct_to_html`` — identical algorithm.
    Returns the original string on failure (no ``<content>`` token or parse error).
    """
    if not otsl_str or not isinstance(otsl_str, str):
        return otsl_str if otsl_str else ""
    if "<content>" not in otsl_str:
        return ""
    try:
        grid, total_cols = _parse_struct_to_grid(otsl_str)
        if not grid:
            return otsl_str
        return _grid_to_html(grid, total_cols)
    except Exception:
        return otsl_str


# ---------------------------------------------------------------------------
# Markdown → HTML table
# ---------------------------------------------------------------------------

def markdown_to_html_table(s: str) -> str:
    if not s:
        return ""
    if "<table" in s.lower():
        return s.strip()
    return markdown.markdown(s, extensions=["tables"]).strip()


# ---------------------------------------------------------------------------
# HTML table validation
# ---------------------------------------------------------------------------

def is_valid_html_table(s: str) -> bool:
    """True iff *s* parses (via lxml) into a real HTML table with rows & cells.

    Stricter than a substring check: bare text or malformed markup that merely
    contains the ``<table`` token is rejected.  Adapters use this to decide
    whether an OTSL/markdown conversion succeeded and emit
    ``ATTR_CONVERSION_FAILED`` when it does not.
    """
    if not s or not s.strip():
        return False
    try:
        doc = lxml.html.fromstring(s)
    except Exception:  # lxml parse errors (ParserError / LxmlError) and friends
        return False
    for tbl in doc.xpath("//table"):
        if tbl.xpath(".//tr") and tbl.xpath(".//td | .//th"):
            return True
    return False
