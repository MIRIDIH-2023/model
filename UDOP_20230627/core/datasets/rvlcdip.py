import json
import logging
import os
import random

from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocCLS

import pandas as pd

EMPTY_BOX = [0, 0, 0, 0]
SEP_BOX = [1000, 1000, 1000, 1000]

logger = logging.getLogger(__name__)


def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def get_bb(bb):
    bbs = [float(j) for j in bb]
    xs, ys = [], []
    for i, b in enumerate(bbs):
        if i % 2 == 0:
            xs.append(b)
        else:
            ys.append(b)
    return [min(xs), min(ys), max(xs), max(ys)]


def get_rvlcdip_labels():
    return [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo"
    ]


class RvlCdipDataset(Dataset):

    #NUM_LABELS = 16

    def __init__(self , xml_sample_loc , json_sum_loc, image_path, tokenizer , data_args , mode='train'):

        """ Structure of data directory:

            --- xml_sample_loc (.csv)
                   ├── images_url
                   └── labels_url
            --- data (folder)
                   └── processed_sample{index} .json
        """
        self.main_df = pd.read_csv(xml_sample_loc) # xml_sample.csv 파일 저장
        self.image_path = image_path

        with open(json_sum_loc, 'r', encoding='utf8') as f:
            self.main_json_data = json.load(f)

        if mode == 'train': #train ,val, test 에 따라 사용하는 data의 범위가 다름. (근데 self-supervised도 이거 필요 있나..? )
            file_data_range = ( 0 , int(len(self.main_df) * 0.6 ) )
        elif mode == 'val':
            file_data_range = ( int(len(self.main_df) * 0.6 ) , int(len(self.main_df) * 0.8 ) )
        elif mode == 'test':
            file_data_range = ( int(len(self.main_df) * 0.8 ) , int(len(self.main_df) ) )
        else:
            raise NotImplementedError

        self.cls_bbox = EMPTY_BOX[:]
        self.pad_bbox = EMPTY_BOX[:]
        self.sep_bbox = SEP_BOX[:]

        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0

        label_list = get_rvlcdip_labels() #classification task에서만 사용하는 변수
        self.label_list = label_list
        self.label_map = dict(zip(list(range(len(self.label_list))), self.label_list))
        self.n_classes = len(label_list)
        self.label_list = label_list

        self.image_size = data_args.image_size

        self.examples = []
        self.labels = []
        self.images = []

        self.cls_collator = DataCollatorForT5DocCLS( #기존에 정의한 토크나이저 선언
                json_data = self.main_json_data ,
                  tokenizer=tokenizer,
            )

        results = [self.load_file(file_idx) for file_idx in tqdm(range(file_data_range[0],file_data_range[1]))]
        for labels, examples, images in results:
            self.labels += labels
            self.examples += examples
            self.images += images
        
        assert len(self.labels) == len(self.examples)

    def load_file(self, file_idx):

        labels = []
        examples = []
        images = []

        labels.append(0) ############### label 미정으로 일단 다 0 ##########
        examples.append(file_idx)
        images.append(file_idx)

        return labels, examples, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index): #완료
        try:
            label = self.labels[index]
            label = self.label_map[int(label)]

            rets, n_split = read_ocr_core_engine(json_data = self.main_json_data , image_path = self.image_path , file_idx = index , tokenizer = self.tokenizer, max_seq_length = self.max_seq_length, num_img_embeds = self.num_img_embeds, image_size = self.image_size)

            if n_split == 0:
                # Something wrong with the .ocr.json file
                print(f"EMPTY ENTRY in index {index}")
                return self[(index + 1) % len(self)]
            for i in range(n_split): #정상적으로 코드 실행됬다면 n_split==1 임.
                text_list, bbox_list, image, page_size = rets[i]
                (width, height) = page_size
                bbox = [  #이미지 크기에 맞게 정규화
                    [
                        b[0] / width,
                        b[1] / height,
                        b[2] / width,
                        b[3] / height,
                    ]
                    for b in bbox_list
                ]

                visual_bbox_input = get_visual_bbox(self.image_size) # (x_min, y_min, x_max, y_max) 형태의 좌표로 이루어진 텐서 반환

                input_ids = self.tokenizer.convert_tokens_to_ids(text_list) #토큰 자른것들을 token id들로 변환

                input_ids, labels, bbox_input = self.cls_collator("user prompt", input_ids, bbox, label) #prompt 붙여서 최종 input,bbox,label을 만듦. ################################
                attention_mask = [1] * len(input_ids)
                decoder_attention_mask = [1] * len(labels)

                char_list = [0]
                char_bbox_list = [[0,0,0,0]]
                char_ids = torch.tensor(char_list, dtype=torch.long)
                char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)

                bbox_input = torch.tensor(bbox_input, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
                assert len(bbox_input) == len(input_ids)
                assert len(bbox_input.size()) == 2
                assert len(char_bbox_input.size()) == 2

                return_dict =  {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "seg_data": bbox_input,
                    "visual_seg_data": visual_bbox_input,
                    "decoder_attention_mask": decoder_attention_mask,
                    "image": image,
                    'char_ids': char_ids,
                    'char_seg_data': char_bbox_input
                }
                assert input_ids is not None

                return return_dict
        except: #오류가 났다는 거는 파일이 없다는 것. 해당 상황에서는 index+1 파일 불러오는 것으로 대체
            #image는 로딩 중 오류 생긴 파일 그냥 해당 index가 없게 저장해서 문제 없음.
            #json파일도 오류 생긴건 해당 index없어서 걸러짐.
            print(f"{index} 파일을 {0}로 대체")

            return self.__getitem__(0)

            #return self[(index + 1) % len(self)]

    #def get_labels(self): # classification에서 label의 종류 출력하는 함수. 우리는 필요 없을 듯.
    #    return list(map(str, list(range(self.NUM_LABELS))))

    def pad_tokens(self, input_ids, bbox): #이건 그냥 길이 max_len에 맞게 맞추는 함수
        # [CLS], sentence, [SEP]
        tokenized_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        start_token, _, end_token = tokenized_tokens[0], tokenized_tokens[1:-1], tokenized_tokens[-1]

        sentence = tokenized_tokens
        expected_seq_length = self.max_seq_length - self.num_img_embeds
        mask = torch.zeros(expected_seq_length)
        mask[:len(sentence)] = 1

        bbox = [self.cls_bbox] + bbox + [self.sep_bbox]
        while len(sentence) < expected_seq_length:
            sentence.append(self.tokenizer.pad_token_id)
            bbox.append(self.pad_bbox)

        assert len(sentence) == len(bbox)
        return (sentence, mask, bbox, start_token, end_token)

# 해당 부분은 json파일의 문장을 line-by-line으로 읽는 것으로 해당 함수 수정 완료.
def read_ocr_core_engine(json_data,image_path, file_idx, tokenizer, max_seq_length=None, num_img_embeds=None, image_size=224):
    #max_seq_length와 num_img_embeds 는 원본 코드에서도 안쓰는데 왜있는거지?

    #file_ 와 image_dir 모두 1 2 3 ... index 임.

    data = json_data[file_idx] # 하나로 합쳐진 json파일에서 현재 idx 가져옴.

    rets = []
    n_split = 0

    #page_size = (1280,720)
    page_size = data['form'][0]['sheet_size']

    tiff_images =  Image.open(f"{image_path}/image_{file_idx}.png")

    image = img_trans_torchvision(tiff_images, image_size)

    text_list, bbox_list = [], []
    for form in data['form']: #문장별로 쪼갬
      for word in form['words']: #단어별로 쪼갬

        if word == ' ': #띄어쓰기는 건너뛰기
          continue

        sub_tokens = tokenizer.tokenize(word['text']) #단어별로 쪼갠걸 다시 토큰화 (하나의 단어도 여러개의 토큰 가능)
        for sub_token in sub_tokens:
          text_list.append(sub_token)
          bbox_list.append(word['box']) #현재는 단어별 bbox, 추후 문장별 bbox로도 수정 가능
          #bbox_list.append(form['box'])

    if len(text_list) > 0:
      rets.append([text_list, bbox_list, image, page_size])

    assert len(text_list) == len(bbox_list)
    n_split = len(rets)

    return rets, n_split




