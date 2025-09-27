import json
import re
import unicodedata
from collections import defaultdict
from typing import Union

import pandas as pd


def normalize_text(s: str, normalize_whitespace: bool = True) -> str:
    """Unicode-normalize and optionally collapse whitespace. Return empty string for None."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    if normalize_whitespace:
        s = re.sub(r"\s+", " ", s)
    return s


def extract_texts_from_json(json_input: Union[str, list]) -> list:
    """
    Accepts either a path to a JSON file or a loaded list-of-dicts.
    Returns list of (text, source_id, location) tuples.
    location is 'question' or 'relevant_questions'.
    """
    if isinstance(json_input, str):
        with open(json_input, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_input

    out = []
    for item in data:
        sid = item.get("id")
        q = item.get("question")
        if q:
            out.append((q, sid, "question"))
        for rel in item.get("relevant_questions", []):
            c = rel.get("content")
            if c:
                out.append((c, sid, "relevant_questions"))
    return out


def compare_datasets(
    json_input: Union[str, list],
    csv_path: str,
    normalize_whitespace: bool = True,
    csv_content_col: str = "content",
) -> pd.DataFrame:
    """
    Compare texts from JSON (question + relevant_questions[].content) to CSV.content.
    Returns a DataFrame of exact matches with counts and where they were found.
    """
    json_texts = extract_texts_from_json(json_input)
    df_csv = pd.read_csv(csv_path, dtype=str)  # read all as strings
    df_csv[csv_content_col] = df_csv[csv_content_col].fillna("")

    # Build maps: normalized_text -> list of (original_text, json_ids, locations)
    json_map = defaultdict(lambda: {"originals": set(), "ids": set(), "locations": []})
    for text, sid, loc in json_texts:
        norm = normalize_text(text, normalize_whitespace)
        json_map[norm]["originals"].add(text)
        json_map[norm]["ids"].add(sid)
        json_map[norm]["locations"].append(loc)

    csv_map = defaultdict(lambda: {"rows": []})
    for _, row in df_csv.iterrows():
        content = row.get(csv_content_col, "")
        norm = normalize_text(content, normalize_whitespace)
        csv_map[norm]["rows"].append(row.to_dict())

    # Intersection: normalized keys present in both maps
    matches = []
    for key in set(json_map.keys()) & set(csv_map.keys()):
        matches.append(
            {
                "matched_text_normalized": key,
                "json_examples": list(json_map[key]["originals"])[:3],
                "json_ids": [x for x in sorted(json_map[key]["ids"]) if x is not None],
                "json_locations": json_map[key]["locations"][:10],
                "csv_count": len(csv_map[key]["rows"]),
                "csv_examples": [
                    r.get(csv_content_col) for r in csv_map[key]["rows"][:3]
                ],
            }
        )

    result_df = pd.DataFrame(matches)
    # order by csv_count desc
    if not result_df.empty:
        result_df = result_df.sort_values("csv_count", ascending=False).reset_index(
            drop=True
        )
    return result_df


# -------------------------
# Example usage:
# -------------------------
# If your JSON is in a file:
matches_df = compare_datasets(
    "dataset_collection/data/evaluation_data.json",
    "dataset_collection/data/raw_es_results.csv",
)
#
# Or if you already have the JSON loaded into a Python variable `dataset1_list`:
# matches_df = compare_datasets(dataset1_list, "dataset2.csv")
#
# Show results:
print(matches_df)
# Save to CSV:
# matches_df.to_csv("text_matches.csv", index=False)
