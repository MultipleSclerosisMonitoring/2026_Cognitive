"""Compare workbook versions without exporting participant identifiers or values."""
import argparse
import hashlib
import json
from pathlib import Path
import xlrd
from restart.audit import audit, SCHEMA


def indexed_rows(book, sheet_index, header):
    sheet = book.sheet_by_index(sheet_index)
    rows = {}
    for row in range(header + 1, sheet.nrows):
        code = str(sheet.cell_value(row, 0)).strip()
        if not code:
            continue
        if code in rows:
            raise ValueError('Duplicate code within sheet: resolve before comparison')
        rows[code] = sheet.row_values(row)
    return rows


def compare(old_path, new_path):
    old_book = xlrd.open_workbook(str(old_path))
    new_book = xlrd.open_workbook(str(new_path))
    result = {'old_sha256': hashlib.sha256(Path(old_path).read_bytes()).hexdigest(),
              'new_audit': audit(new_path), 'changes': []}
    new_codes = []
    for sheet_index, header, group, columns in SCHEMA:
        old = indexed_rows(old_book, sheet_index, header)
        new = indexed_rows(new_book, sheet_index, header)
        old_keys, new_keys = set(old), set(new)
        common = old_keys & new_keys
        new_codes.append(new_keys)
        changes = {}
        for variable, col in columns.items():
            changes[variable] = sum(old[key][col] != new[key][col] for key in common)
        sheet = new_book.sheet_by_index(sheet_index)
        outcomes = [col for name, col in columns.items() if name != 'edad']
        uncoded = sum(not str(sheet.cell_value(i, 0)).strip() and
                      any(sheet.cell_value(i, col) != '' for col in outcomes)
                      for i in range(header + 1, sheet.nrows))
        result['changes'].append({'group': group, 'added_codes': len(new_keys-old_keys),
            'removed_codes': len(old_keys-new_keys), 'common_codes': len(common),
            'changed_cells_among_common_codes': changes,
            'uncoded_rows_with_audited_outcomes': uncoded})
    result['cross_sheet_code_overlap'] = len(new_codes[0] & new_codes[1])
    result['age_validation_passed'] = all(
        g['variables']['edad']['nonpositive_numeric'] == 0 and
        g['variables']['edad']['non_numeric_or_nonfinite'] == 0
        for g in result['new_audit']['groups'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f'Comparación agregada guardada en {args.output}')
