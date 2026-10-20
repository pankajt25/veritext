# Web scraper and tabular dataset processor
import csv
import json
from pathlib import Path

def process_csv_records(input_path, output_json):
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items()}
            records.append(clean_row)

    Path(output_json).write_text(json.dumps(records, indent=2))
    return len(records)
