import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
from transformers import PreTrainedTokenizerBase

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
        
        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

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

        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list