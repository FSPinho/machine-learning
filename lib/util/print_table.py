from typing import List, Union


def print_table(table: Union[List[dict], List[List]], max_column_width=64):
    if len(table) == 0:
        return

    headers = table[0].keys() if isinstance(table[0], dict) else list(range(len(table[0])))
    headers_widths = {
        header: min(max(*map(len, [
            _get_cell_output(row[header], None, type(row[header]))
            for row in table
        ]), len(str(header))), max_column_width)
        for header in headers
    }
    headers_types = {
        header: type(table[0][header])
        for header in headers
    }

    rows_output = [_get_row_output(row, headers, headers_widths, headers_types) for row in table]
    body_output = "".join(rows_output)
    header_output = _get_header_output(headers, headers_widths, headers_types)
    footer_output = _get_footer_output(headers_widths)
    output = header_output + body_output + footer_output
    print(output)


def _get_row_output(row, headers, headers_widths, headers_types):
    cells = [_get_cell_output(row[header], headers_widths[header], headers_types[header]) for header in headers]
    output = " │ ".join(cells)
    return f"│ {output} │\n"


def _get_header_output(headers, headers_widths, header_types):
    cells = [_get_cell_output(header, headers_widths[header], header_types[header]) for header in headers]
    header_output = " │ ".join(cells)
    header_output = f"│ {header_output} │"
    top_output = "─┬─".join(["─" * width for width in headers_widths.values()])
    top_output = f"┌─{top_output}─┐"
    bottom_output = "─┼─".join(["─" * width for width in headers_widths.values()])
    bottom_output = f"├─{bottom_output}─┤"
    return f"{top_output}\n{header_output}\n{bottom_output}\n"


def _get_footer_output(headers_widths):
    footer_output = "─┴─".join(["─" * width for width in headers_widths.values()])
    return f"└─{footer_output}─┘"


def _get_cell_output(value, width, _type):
    width = width or ""
    alignment = ">" if _type in (int, float) else "<"

    if isinstance(value, (list, tuple)):
        value = ", ".join(map(str, value))

    if isinstance(value, int):
        template = f"{{value:{alignment}{width}}}"
    elif isinstance(value, float):
        template = f"{{value:{alignment}{width}.3f}}"
    else:
        template = f"{{value:{alignment}{width}}}"

    output = template.format(value=value)

    if width and len(output) > width:
        output = f"{output[:width - 3]}..."

    return output
