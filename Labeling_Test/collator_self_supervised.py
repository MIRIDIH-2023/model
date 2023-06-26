import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
from transformers import PreTrainedTokenizerBase

# TODO: 각각의 Task에 따라 라벨링 하기
# 근데 Dataset에서 input text가 id로 변환되어가지고
# 개인적인 생각으로는 다시 token으로 변환하던가, 아니면 decode까지 해서 하던가 해야 할 듯??
# 그냥 Dataset에서 따로 라벨링을 하는 게 좀 더 편할 것 같긴 한디...

# Wrapper Class
# User Prompt에 따라서 Collator를 호출해주는 클래스입니다
class DataCollatorForSelfSupervisedTasks:

    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        
        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.LM = DataCollatorForT5LayoutModeling(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.VT = DataCollatorForT5VisTextRec(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.JR = DataCollatorForT5JointReconstruction(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )


    def __call__(self, user_prompt, ori_input_ids, ori_bbox_list):

        if 'Layout Modeling' in user_prompt:
            return self.LM(user_prompt, ori_input_ids, ori_bbox_list)
        
        elif 'Visual Text Recognition' in user_prompt:
            return self.VT(user_prompt, ori_input_ids, ori_bbox_list)
        
        elif 'Joint Text-Layout Reconstruction' in user_prompt:
            return self.JR(user_prompt, ori_input_ids, ori_bbox_list)
        
        else:
            raise ValueError("Invalid user prompt")


class DataCollatorForT5LayoutModeling:
    """
    Data collator used for T5 Layout Modeling
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        
        # prompt_text = user_prompt   # 'Layout Modeling'
        # prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # input_ids = prompt_ids + ori_input_ids
        # bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list
        
        # input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        # labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        
        # input: "The cute dog walks in the park"
        # 일단 token화 까지 하자!
        # 알고리즘=> input에서 "<extra_l_id_0> The cute dog </extra_l_id_0> walks in the park" 으로 masking: 0.75 (75%)
        # label: <extra_l_id_0><loc_100><loc_200><loc_400><loc_500> => 위치는 masking한 단어의 bbox
        # issue: <extra_l_id_0>의 bbox는?, The cute dog의 bbox를 포함해야 하는가? 
        # Some text token like manually crafted prompts have no locations. So, we set their layout bounding boxes to be (0, 0, 0, 0)
        # 위는 논문에서 말한 내용을 참고했을 때 <extra_l_id_0>의 bbox는 (0, 0, 0, 0)으로 설정해야할 듯
        # The cute dog의 bbox는 포함하는 쪽으로 결정: 혹시 모르니까 길이는 맞춰주어야 할 것 같아서
        # The cute dog는 text list에 한번에 안 넣고 각각 The, cute, dog 씩 넣고 token화 한다.
        # label 설정만 신경쓰면 될 듯
        
        
        # TODO: 라벨링 하기 (Layout_Modeling_Test.py 참고하면서)
        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list


class DataCollatorForT5VisTextRec:
    """
    Data collator used for T5 Visual Text Recognition
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box

        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list


class DataCollatorForT5JointReconstruction:
    """
    Data collator used for T5 Joint Text-Layout Reconstruction
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" listb
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box

        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        # TODO: 라벨링 하기 (Joint_Text_Layout_Reconstruction_Test.py 참고하면서)
        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list