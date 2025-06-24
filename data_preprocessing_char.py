import os
import json
import unicodedata
from collections import defaultdict

input_dir = './data/vnondb/InkData_word_train_segmented'
output_dir = './dataset/vnondb/InkData_word_train_segmented'

# Create the output directory (flat)
os.makedirs(output_dir, exist_ok=True)

# Counter for unique Unicode name-based filenames
substring_counter = defaultdict(int)

def safe_unicode_name(char: str) -> str:
    """
    Converts a character to a flat, filesystem-safe Unicode name.
    Examples:
        'A' → 'latin_capital_letter_a'
        ' ' → 'space'
        '#' → 'number_sign'
    """
    char = char.strip()
    if not char:
        return "space"

    if len(char) == 1:
        try:
            return unicodedata.name(char).lower().replace(" ", "_").replace("-", "_")
        except ValueError:
            return f"char_{ord(char)}"
    elif len(char) > 1:
        # Handle multi-character substrings like "AB" or "the"
        return "multi_" + "_".join(safe_unicode_name(c) for c in char)

    return char

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get('gt_segmentation', {}).get('segments', [])
        points = data.get('points', [])
        points_by_stroke = defaultdict(list)
        for pt in points:
            points_by_stroke[pt['stroke']].append(pt)

        for seg in segments:
            substring = seg.get('substring', '')
            ink_ranges = seg.get('inkRanges', [])
            strokes = set()
            substring_points = []

            for ir in ink_ranges:
                strokes.update([ir['startStroke'], ir['endStroke']])
                for stroke_num in range(ir['startStroke'], ir['endStroke'] + 1):
                    stroke_points = points_by_stroke.get(stroke_num, [])
                    start_idx = ir['startPoint'] if stroke_num == ir['startStroke'] else 0
                    end_idx = ir['endPoint'] if stroke_num == ir['endStroke'] else len(stroke_points) - 1
                    substring_points.extend(stroke_points[start_idx:end_idx + 1])

            if ink_ranges:
                segment_data = {
                    'file': filename,
                    'substring': substring,
                    'inkRanges': ink_ranges,
                    'strokes': sorted(list(strokes)),
                    'points': substring_points
                }

                # 🆕 Use Unicode name directly as filename prefix
                safe_substring = safe_unicode_name(substring)
                idx = substring_counter[safe_substring]
                substring_counter[safe_substring] += 1

                out_name = f"{safe_substring}_{idx}.json"
                out_path = os.path.join(output_dir, out_name)

                with open(out_path, 'w', encoding='utf-8') as out_f:
                    json.dump(segment_data, out_f, indent=2)

                # 🌟 Optional: debug print
                # print(f"[SAVED] '{substring}' -> {out_name}")