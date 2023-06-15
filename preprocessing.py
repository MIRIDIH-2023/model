# coding=utf-8
# Code to preprocess sample CSV and XML to JSON
# Requirements: Download sample CSV file from slack and save it as data/metadata.csv

import json
import math
from io import BytesIO

import pandas as pd
import requests
import xml2dict
from PIL import Image


def process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, angle):
    RATIO = IM_SIZE[0] / SHEET_SIZE[0]
    x1, y1, x2, y2 = map(float, XML_BBOX)
    x1, y1, x2, y2 = (x1 * RATIO, y1 * RATIO, x2 * RATIO, y2 * RATIO)

    if angle != 0:
        angle = math.radians(angle)
        # Calculate the center point of the bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Calculate the distance from the center to each corner of the bbox
        distance_x = (x1 - center_x)
        distance_y = (y1 - center_y)
        # Apply rotation to the distances
        new_distance_x = distance_x * math.cos(angle) - distance_y * math.sin(angle)
        new_distance_y = distance_x * math.sin(angle) + distance_y * math.cos(angle)
        # Calculate the new corners after rotation
        x1 = center_x + new_distance_x
        y1 = center_y + new_distance_y
        x2 = center_x - new_distance_x
        y2 = center_y - new_distance_y
    
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    return x1, y1, x2, y2


def main():
    # Read sample CSV and download thumbnail, XML
    df = pd.read_csv("data/metadata.csv")
    sample_sheet = df.iloc[0]

    sample_thumbnail = Image.open(BytesIO(requests.get(sample_sheet["thumbnail_url"]).content))
    sample_xml = requests.get(sample_sheet["sheet_url"]).content.decode("utf-8")
    sample_json = xml2dict.parse(sample_xml)

    processed_json = {}
    processed_json['form'] = []

    SHEET_SIZE = tuple(map(int, sample_json['SHEET']['SHEETSIZE'].values()))
    IM_SIZE = sample_thumbnail.size

    # Process XML to json
    for i, texts in enumerate(sample_json['SHEET']['TEXT']):
        XML_BBOX = tuple(texts['Position'].values())

        text = texts['Text'].replace("\u00a0", "") # Remove non-breaking space
        x1, y1, x2, y2 = process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, int(texts['@Rotate']))

        processed_json['form'].append({
            "text": text,
            "box": [x1, y1, x2, y2],
            "id": i
        })

    json.dump(processed_json, open("data/processed_sample.json", "w"), indent=4)


if __name__ == "__main__":
    main()
