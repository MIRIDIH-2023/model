import json
import math
from io import BytesIO

import pandas as pd
import requests
import xml_to_dict
import json
from PIL import Image


def process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, angle, center):
    RATIO = IM_SIZE[0] / SHEET_SIZE[0]
    x1, y1, x2, y2 = map(float, XML_BBOX)
    x1, y1, x2, y2 = (x1 * RATIO, y1 * RATIO, x2 * RATIO, y2 * RATIO)
    center = (center[0] * RATIO, center[1] * RATIO)

    if angle != 0:
        angle = 360 - angle
        angle = math.radians(angle)
        # Calculate the center point of the bbox
        center_x, center_y = center
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

def get_render_bbox(text):
    if text['RenderPos'] == None:
        return []
    render_pos = json.loads(text['RenderPos'])
    render_bbox = []

    left, top, right, bottom = map(float, text['Position'].values())

    for render in render_pos['c']:
        x, a, w, y = map(float, [render['x'], render['a'], render['w'], render['y']])
        left_ = left + x
        right_ = left_ + w
        bottom_ = top + y
        top_ = bottom_ - a

        render_bbox.append((left_, top_, right_, bottom_))

    return render_bbox

def get_bbox(render_bbox):
    min_x = min([x[0] for x in render_bbox])
    min_y = min([x[1] for x in render_bbox])
    max_x = max([x[2] for x in render_bbox])
    max_y = max([x[3] for x in render_bbox])

    return min_x, min_y, max_x, max_y

def process_xml_dict(xml_dict, thumbnail):
    processed_json = {}
    processed_json['form'] = []

    SHEET_SIZE = tuple(map(int, xml_dict['SHEET']['SHEETSIZE'].values()))
    IM_SIZE = thumbnail.size

    # Process XML to json
    for i, text in enumerate(xml_dict['SHEET']['TEXT']):
        left, top, right, bottom = map(float, text['Position'].values())
        center = ((left + right) / 2, (top + bottom) / 2)

        render_bbox = get_render_bbox(text)
        if len(render_bbox) == 0: continue

        XML_BBOX = get_bbox(render_bbox)

        t = text['Text']
        x1, y1, x2, y2 = process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, int(text['@Rotate']), center)

        processed_json['form'].append({
            "text": t,
            "box": [x1, y1, x2, y2],
            "font_id": int(text['Font']['@FamilyIdx']),
            "font_size": float(text['Font']['@Size']),
            "style": {
                "bold": text['Font']['Style']['@Bold'] == 'true',
                "italic": text['Font']['Style']['@Italic'] == 'true',
                "strikeout": text['Font']['Style']['@Strikeout'] == 'true',
                "underline": text['Font']['Style']['@Underline'] == 'true'
            },
            "linespace": float(text['Font']['@LineSpace']),
            "opacity": float(text['@Opacity']),
            "rotate": float(text['@Rotate']),
            "id": i
        })

        processed_json['form'][-1]['words'] = []

        render_pos = json.loads(text['RenderPos'])

        for j, bbox in enumerate(render_bbox):
            x1_, y1_, x2_, y2_ = process_bbox(bbox, IM_SIZE, SHEET_SIZE, int(text['@Rotate']), center)
            color = render_pos['c'][j]['f']
            color = color[4:-1]
            color = list(map(int, color.split(",")))
            processed_json['form'][-1]['words'].append({
                "text": render_pos['c'][j]['t'],
                "box": [x1_, y1_, x2_, y2_],
                "font_size": float(render_pos['c'][j]['s']),
                "letter_spacing": float(render_pos['c'][j]['ds']),
                "font_id": int(render_pos['c'][j]['yd']),
                "color": color
            })

    return processed_json

def process_xml(sheet_url, thumbnail_url):
    sample_thumbnail = Image.open(BytesIO(requests.get(thumbnail_url).content))
    sample_xml = requests.get(sheet_url).content.decode("utf-8")
    sample_json = xml_to_dict.XMLtoDict().parse(sample_xml)

    processed_json = process_xml_dict(sample_json, sample_thumbnail)

    return processed_json

def main():
    # Read sample CSV and download thumbnail, XML
    df = pd.read_csv("data/metadata.csv")
    sample_sheet = df.iloc[0]

    processed_json = process_xml(sample_sheet['sheet_url'], sample_sheet['thumbnail_url'])

    json.dump(processed_json, open("data/processed_sample.json", "w"), indent=4)


if __name__ == "__main__":
    main()
