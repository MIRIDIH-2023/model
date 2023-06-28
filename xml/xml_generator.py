from lxml import etree
from typing import List, Dict, Tuple

def get_SHEET(SHEET_attrs: Dict[str, str] = {}) -> etree._Element:
    SHEET = etree.Element('SHEET')
    SHEET.attrib.update(SHEET_attrs)
    return SHEET

def get_SHEETSIZE(size: Tuple[int, int]) -> etree._Element:
    width, height = size

    SHEETSIZE = etree.Element('SHEETSIZE')
    SHEETSIZE.set('cx', str(width))
    SHEETSIZE.set('cy', str(height))
    return SHEETSIZE

def get_TEMPLATE(size: Tuple[int, int], TEMPLATE_attrs: Dict[str, str] = {}) -> etree._Element:
    width, height = size
    
    TEMPLATE = etree.Element('TEMPLATE')
    TEMPLATE.set('Width', str(width))
    TEMPLATE.set('Height', str(height))
    TEMPLATE.attrib.update(TEMPLATE_attrs)
    return TEMPLATE

def get_BACKGROUND(color: str, BACKGROUND_attrs: Dict[str, str] = {}) -> etree._Element:
    BACKGROUND = etree.Element('BACKGROUND')
    BACKGROUND.set('Color', color)
    BACKGROUND.set('LayerName', '')
    BACKGROUND.attrib.update(BACKGROUND_attrs)

    PureSkinSize = etree.Element('PureSkinSize')
    PureSkinSize.set('cx', '0')
    PureSkinSize.set('cy', '0')
    BACKGROUND.append(PureSkinSize)

    CropRect = etree.Element('CropRect')
    CropRect.set('Left', '0')
    CropRect.set('Top', '0')
    CropRect.set('Right', '0')
    CropRect.set('Bottom', '0')
    BACKGROUND.append(CropRect)

    return BACKGROUND

def get_GUIDELINES(GUIDELINES_attrs: Dict[str, str] = {}) -> etree._Element:
    GUIDELINES = etree.Element('GUIDELINES', **GUIDELINES_attrs)

    GUIDELINE = etree.Element('GUIDELINE')
    GUIDELINE.set('InitPosition', '0')
    GUIDELINE.set('LastPosition', '0')
    
    HORIZONTAL = etree.Element('HORIZONTAL')
    HORIZONTAL.set('CurrentPosition', '0')
    HORIZONTAL.set('Positive', 'true')
    HORIZONTAL.append(GUIDELINE)

    VERTICAL = etree.Element('VERTICAL')
    VERTICAL.set('CurrentPosition', '0')
    VERTICAL.set('Positive', 'true')
    VERTICAL.append(GUIDELINE)

    GUIDELINES.append(HORIZONTAL)
    GUIDELINES.append(VERTICAL)

    return GUIDELINES

def get_PageAnimations(PageAnimations_attrs: Dict[str, str] = {}) -> etree._Element:
    PageAnimations = etree.Element('PageAnimations', **PageAnimations_attrs)
    PageAnimations.set('pageAnimationPreset', 'NONE')
    return PageAnimations

def get_TEXT(
		text: str, 
		position: Tuple[int, int, int, int],
		TEXT_attrs: Dict[str, str] = {},
		Position_attrs: Dict[str, str] = {},
		Hyperlink_attrs: Dict[str, str] = {},
		LogData_attrs: Dict[str, str] = {},
		Text_attrs: Dict[str, str] = {},
		Font_attrs: Dict[str, str] = {},
		Style_attrs: Dict[str, str] = {},
		Curve_attrs: Dict[str, str] = {},
		Effect_attrs: Dict[str, str] = {},
		Outline_attrs: Dict[str, str] = {},
		Shadow_attrs: Dict[str, str] = {},
		Fill_attrs: Dict[str, str] = {},
		) -> etree._Element:
	Left, Top, Right, Bottom = position

	TEXT = etree.Element('TEXT')
	TEXT.set('Rotate', '0')
	TEXT.set('Opacity', '255')
	TEXT.set('FlipH', '0')
	TEXT.set('FlipV', '0')
	TEXT.set('AddedBy', '0')
	TEXT.set('GroupId', '')
	TEXT.set('LayerName', '')
	TEXT.set('Priority', '5')
	TEXT.set('NewVersion', 'true')
	TEXT.set('Alignment', '1')
	TEXT.set('VAlignment', '0')
	TEXT.set('UpperCase', 'false')
	TEXT.set('Renderable', 'true')
	TEXT.set('IsPageNumberItem', 'false')
	TEXT.set('Type', '')
	TEXT.attrib.update(TEXT_attrs)

	Position = etree.Element('Position')
	Position.set('Left', str(Left))
	Position.set('Top', str(Top))
	Position.set('Right', str(Right))
	Position.set('Bottom', str(Bottom))
	Position.attrib.update(Position_attrs)

	Hyperlink = etree.Element('Hyperlink')
	Hyperlink.set('Active', 'false')
	Hyperlink.set('Type', 'URL')
	Hyperlink.set('Value', '')
	Hyperlink.attrib.update(Hyperlink_attrs)

	LogData = etree.Element('LogData')
	LogData.set('RefResourceKey', '')
	LogData.set('RefResourceType', '')
	LogData.attrib.update(LogData_attrs)

	Text = etree.Element('Text')
	Text.text = text
	Text.attrib.update(Text_attrs)

	Font = etree.Element('Font')
	Font.set('Color', 'FF000000')
	Font.set('Family', 'M PLUS 2 Bold')
	Font.set('FamilyIdx', '5524')
	Font.set('Size', '66')
	Font.set('LineSpace', '1.2')
	Font.attrib.update(Font_attrs)

	Style = etree.Element('Style')
	Style.set('Bold', 'false')
	Style.set('Italic', 'false')
	Style.set('Strikeout', 'false')
	Style.set('Underline', 'false')
	Style.attrib.update(Style_attrs)

	Font.append(Style)

	Curve = etree.Element('Curve')
	Curve.set('IsCurved', 'false')
	Curve.set('Clockwise', 'true')
	Curve.attrib.update(Curve_attrs)

	Effect = etree.Element('Effect')
	Effect.set('TextSpace', '0')
	Effect.set('ScaleX', '1')
	Effect.attrib.update(Effect_attrs)

	Outline = etree.Element('Outline')
	Outline.set('DoOutline', 'false')
	Outline.set('Color', '0000000')
	Outline.set('Size', '5')
	Outline.attrib.update(Outline_attrs)

	Shadow = etree.Element('Shadow')
	Shadow.set('DoShadow', 'false')
	Shadow.set('Color', '28000000')
	Shadow.set('Distance', '10')
	Shadow.set('Angle', '315')
	Shadow.set('Spread', '0')
	Shadow.attrib.update(Shadow_attrs)

	Fill = etree.Element('Fill')
	Fill.set('Type', 'COLOR')
	Fill.attrib.update(Fill_attrs)

	Effect.append(Outline)
	Effect.append(Shadow)
	Effect.append(Fill)

	TEXT.append(Position)
	TEXT.append(Hyperlink)
	TEXT.append(LogData)
	TEXT.append(Text)
	TEXT.append(Font)
	TEXT.append(Curve)
	TEXT.append(Effect)

	return TEXT

def get_default_tags(size: Tuple[int, int] = (1920, 1080), bg_color: str = 'FFFFFFFF') -> List[etree._Element]:
	default_tags = []
	default_tags.append(get_SHEETSIZE(size))
	default_tags.append(get_TEMPLATE(size))
	default_tags.append(get_BACKGROUND(bg_color))
	default_tags.append(get_GUIDELINES())
	default_tags.append(get_PageAnimations())
	return default_tags

if __name__ == '__main__':
  SHEET = get_SHEET()

  default_tags = get_default_tags((1920, 1080), bg_color='FFFFFFFF')
  for tag in default_tags:
    SHEET.append(tag)

  SHEET.append(get_TEXT('Hello World', (0, 0, 100, 100)))
  SHEET.append(get_TEXT('My name is', (40, 40, 100, 100)))
  SHEET.append(get_TEXT('Kazuma', (60, 60, 100, 100), Font_attrs={'size': '93'}))

  xml = etree.ElementTree(SHEET)
  xml.write('test.xml')
