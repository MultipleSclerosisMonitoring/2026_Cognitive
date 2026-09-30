"""Auditoría agregada del libro original; nunca exporta identificadores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xlrd

SCHEMA = [
    (0, 2, 'pacientes', {'edad': 12, 'sdmt_papel': 36, 'sdmt_digital': 34,
                        'tmt_a_papel': 30, 'tmt_b_papel': 31}),
    (1, 1, 'controles', {'edad': 10, 'sdmt_papel': 30, 'sdmt_digital': 28,
                        'tmt_a_papel': 24, 'tmt_b_papel': 25}),
]

def audit(path):
    book = xlrd.open_workbook(str(path))
    result = {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'groups': []}
    for sheet_index, header, group, columns in SCHEMA:
        sheet = book.sheet_by_index(sheet_index)
        if sheet.cell_value(header, 0).strip() != 'Código':
            raise ValueError('Estructura inesperada: revisar el diccionario antes de continuar')
        for name, col in columns.items():
            label = str(sheet.cell_value(header, col if name != 'tmt_b_papel' else col-1)).upper()
            expected = 'EDAD' if name == 'edad' else 'SDMT' if name.startswith('sdmt') else 'TMT'
            if expected not in label:
                raise ValueError(f'Columna inesperada: {group}/{name}')
        rows = [i for i in range(header+1, sheet.nrows) if str(sheet.cell_value(i, 0)).strip()]
        codes = [str(sheet.cell_value(i, 0)).strip() for i in rows]
        entry = {'group': group, 'sheet_index': sheet_index, 'header_excel_row': header+1,
                 'coded_rows': len(rows), 'unique_codes': len(set(codes)), 'variables': {}}
        for name, col in columns.items():
            values = [sheet.cell_value(i, col) for i in rows]
            numeric = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            missing = sum(v == '' for v in values)
            entry['variables'][name] = {'excel_column_index_zero_based': col,
                'numeric': len(numeric), 'empty': missing,
                'non_numeric_or_nonfinite': len(values)-missing-len(numeric),
                'nonpositive_numeric': sum(v <= 0 for v in numeric)}
        result['groups'].append(entry)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/audit.json'))
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(f'Auditoría agregada guardada en {args.output}')
