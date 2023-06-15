# coding=utf-8
# Code to preprocess sample CSV and XML to JSON

import json

import pandas as pd
import requests
import xml2dict

def main():
    # Read sample CSV and download XML and thumbnail
    df = pd.read_csv("data/metadata.csv")
    sample_sheet = df.iloc[0]

    sample_xml = requests.get(sample_sheet["sheet_url"]).content.decode("utf-8")
    sample_json = xml2dict.parse(sample_xml)

    processed_json = {}
    processed_json['form'] = []

    # Process XML to json
    for i, texts in enumerate(sample_json['SHEET']['TEXT']):
        text = texts['Text'].replace("\u00a0", "") # Remove non-breaking space
        x1, y1, x2, y2 = texts['Position'].values()
        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

        processed_json['form'].append({
            "text": text,
            "box": [x1, y1, x2, y2],
            "id": i
        })

    json.dump(processed_json, open("data/processed_sample.json", "w"), indent=4)


if __name__ == "__main__":
    main()
