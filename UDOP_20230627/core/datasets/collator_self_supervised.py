import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

# TODO: 각각의 Task에 따라 라벨링 하기
# 근데 Dataset에서 input text가 id로 변환되어가지고
# 개인적인 생각으로는 다시 token으로 변환하던가, 아니면 decode까지 해서 하던가 해야 할 듯??
# 그냥 Dataset에서 따로 라벨링을 하는 게 좀 더 편할 것 같긴 한디...


def random_masking(L=4096, mask_ratio=0.75):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(L)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=0)

    # keep the first subset
    ids_keep = ids_shuffle[:len_keep]
    ids_remove = ids_shuffle[len_keep:]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([L])
    mask[:len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=0, index=ids_restore)
    print(f'mask = {mask}')
    return mask, ids_restore, ids_remove, ids_keep

# Argument : random_masking의 mask
# Returns : masking할 곳의 slice들
def group_tokens(mask):

    group_lst = []
    i=0
    prev=0

    for m in mask:
        if m == 0:
            if i == prev:
                pass
            else:
                group_lst.append([prev, i])
            prev = i+1
        i += 1

    if prev != i:
        group_lst.append([prev, i])

    print(f'group list : {group_lst}')
    return group_lst

# Argument : ori_bbox_lst, group_tokens의 리턴 list (slices)
# Returns : masking된 부분의 그룹화된 bbox
def group_bbox(bbox_lst, group_lst):

    bbox_group_lst = []

    for s in group_lst:
        target = bbox_lst[s[0]:s[1]]
        if len(target) == 1:
            bbox_group_lst.append(*target)
        else:
            t = target[0][0]
            l = target[0][1]
            b = target[0][2]
            r = target[0][3]
            for i in target[1:]:
                if i[0] < t:
                    t = i[0]
                if i[1] < l:
                    l = i[1]
                if i[2] > b:
                    b = i[2]
                if i[3] > r:
                    b = i[3]
            bbox_group_lst.append([t,l,b,r])
    
    return bbox_group_lst

# Argument : ori_bbox_list, mask_ratio
# Returns : token slices to be masked, grouped bboxes
def mask_process(bbox_list, mask_ratio=0.75):
    l = len(bbox_list)
    mask = random_masking(L=l, mask_ratio=mask_ratio)
    return group_tokens(mask[0]), group_bbox(bbox_list, group_tokens(mask[0]))


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

    # Label Numbering 변수 추가.
    # label_numbering 변수가 하는 일
    #   1. Sentinel Token에 번호를 부여한다
    #   2. 엥 근데 이렇게 구현하면 User_Prompt가 여러 개 포함되는 거 아님?
    #   3. label_numbering에 0이 없으면 user_prompt를 안넣으면 된다.
    # label_numbering 변수의 형태 = [a, b]
    # a = 시작, b = 끝
    # mask_process는 read_ocr_core_engine에서 불러서 mask할 범위를 집어 줄 것이다
    def __call__(self, user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering):

        if 'Layout Modeling' in user_prompt:
            return self.LM(user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering)
        
        elif 'Visual Text Recognition' in user_prompt:
            return self.VT(user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering)
        
        elif 'Joint Text-Layout Reconstruction' in user_prompt:
            return self.JR(user_prompt, ori_input_ids, bbox_list, group_list, ori_bbox_list, label_numbering)
        
        else:
            raise ValueError("Invalid user prompt")

# Sub Class에서 해야 할 일
# 1. Argument를 user_prompt, ori_input_ids, group_list, ori_bbox_list, label_numbering을 받는다.
# 2. label_numbering의 시작이 0이라면, user_prompt를 앞에 id로 변환해서 붙인다. (input_ids에)
# 3. label_numbering에 따라서 sentinel token을 ori_bbox_list를 보면서 붙인다. (input_ids에)
# 4. labeling은 group_list와 ori_bbox_list를 보면서 붙인다 (labels에)
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

    def __call__(self, user_prompt , input_ids, bbox_list, group_list, group_bbox_list, label_numbering, page_size):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        
        res_input_ids = []
        prompt_text = user_prompt
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        res_input_ids += prompt_ids
        res_bbox_list = [[0,0,0,0]] * len(prompt_ids)

        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_l_id_{label_numbering[i]}>', add_special_tokens=True)
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][0]/page_size[0])}>')
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][1]/page_size[1])}>')
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][2]/page_size[0])}>')
            labels += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[i][3]/page_size[1])}>')
            
        slice_pointer=0
        L = len(group_list)
        for i in range(len(input_ids)):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                temp_ids = self.tokenizer.encode(f'<extra_l_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)
                res_input_ids += temp_ids
                res_input_ids.append(input_ids[i])
                res_bbox_list += [[0,0,0,0]] * len(temp_ids)
                res_bbox_list.append(bbox_list[i])
            elif slice_pointer < L and i == group_list[slice_pointer][1] :
                temp_ids = self.tokenizer.encode(f'</extra_l_id{label_numbering[slice_pointer]}>', add_special_tokens=True)
                res_input_ids += temp_ids
                res_input_ids.append(input_ids[i])
                res_bbox_list += [[0,0,0,0]] * len(temp_ids)
                res_bbox_list.append(bbox_list[i])
                slice_pointer += 1
            else:
                res_input_ids.append(input_ids[i])
                res_bbox_list.append(bbox_list[i])
                
        res_input_ids += [1] # </s> token
        res_bbox_list.append([0,0,0,0])
        labels += [1] # </s> token
        
        return res_input_ids, labels, res_bbox_list

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

    # Sub Class에서 해야 할 일
    # 1. Argument를 user_prompt, ori_input_ids, group_list, ori_bbox_list, label_numbering을 받는다.
    # 2. label_numbering의 시작이 0이라면, user_prompt를 앞에 id로 변환해서 붙인다. (input_ids에)
    # 3. label_numbering에 따라서 sentinel token을 ori_bbox_list를 보면서 붙인다. (input_ids에)
    # 4. labeling은 group_list와 ori_bbox_list를 보면서 붙인다 (labels에)
    def __call__(self, user_prompt , input_ids, bbox_list, group_list, group_bbox_list, label_numbering, page_size):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box

        prompt_text = user_prompt
        if label_numbering[0] == 0:
            prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
            tmp_input_ids = prompt_ids
        else:
            tmp_input_ids = []
        tmp_bbox_list = [[0,0,0,0]] * len(prompt_ids)

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_t_id_{label_numbering[i]}>', add_special_tokens=True)[:-1]
            labels += input_ids[group_list[i][0]:group_list[i][1]]


        slice_pointer=0
        for i in range(len(input_ids)):
            if i == group_list[slice_pointer][0]:
                tmp_input_ids += self.tokenizer.encode(f'<extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                tmp_bbox_list.append([0,0,0,0])
                bbox_ids = []
                for j in range(4):
                    if j % 2 == 0:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[0])}>', add_special_tokens=True)[:-1]
                    else:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[1])}>', add_special_tokens=True)[:-1]
                        tmp_bbox_list.append([0,0,0,0])
                tmp_input_ids += bbox_ids
                tmp_input_ids += self.tokenizer.encode(f'</extra_t_id{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                tmp_bbox_list.append([0,0,0,0])
                i = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_bbox_list.append(bbox_list[i])

        tmp_input_ids += [1] # </s> token
        tmp_bbox_list.append([0,0,0,0])
        labels += [1] # </s> token
        return tmp_input_ids, labels, tmp_bbox_list


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

    def __call__(self, user_prompt ,ori_input_text, ori_bbox_list):

        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        masked_input_text, labels, bbox = self.random_masking(ori_input_text, ori_bbox_list, mask_ratio=0.15)
        masked_input_ids = self.tokenizer.encode(masked_input_text, add_special_tokens=True)
        input_ids = prompt_ids + masked_input_ids
        bbox_list = [[0, 0, 0, 0]] * len(prompt_ids) + bbox + [[0, 0, 0, 0]] # <\s> token
        label_ids = self.tokenizer.encode(labels, add_special_tokens=True)

        assert len(input_ids) == len(bbox_list)
        return input_ids, label_ids, bbox_list
        
        
    def random_masking(self, input_text, bbox_list, mask_ratio=0.15):
        idx = 0
        
        total_input_ids, total_targets, total_bbox = [], "", []

        for text,bbox in zip(input_text, bbox_list):
            masked_input_ids, targets, part_bbox = [], "", []

            L = len(text)
            len_keep = int(L * (1 - mask_ratio))

            noise = torch.rand(L)  # Noise in [0, 1]

            # Sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=0)  # Ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=0)

            # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([L])
            mask[:len_keep] = 0
            # Unshuffle to get the binary mask
            mask = torch.gather(mask, dim=0, index=ids_restore)
            mask[0] = 1

            # Convert masked tokens to the '<extra_ids_idx>' format
            is_previous_masked = False
            previous_bbox = None

            for index, (token, token_bbox, is_masked) in enumerate(zip(text, bbox, mask)):
                if is_masked:
                    if not is_previous_masked:
                        masked_input_ids.append(f"<extra_id_{idx}>")
                        tokenized_bbox = list(map(int, [i * self.tokenizer._loc_extra_ids for i in token_bbox]))
                        targets += f" <extra_id_{idx}> {token}" 
                        part_bbox.append([0, 0, 0, 0])
                        idx += 1

                    else: # Previous masked
                        targets += f" {token}"
                        tokenized_bbox = list(map(int, [i * self.tokenizer._loc_extra_ids for i in token_bbox]))
                        tokenized_bbox = [min(previous_bbox[0], tokenized_bbox[0]), min(previous_bbox[1], tokenized_bbox[1]), max(previous_bbox[2], tokenized_bbox[2]), max(previous_bbox[3], tokenized_bbox[3])]

                    previous_bbox = tokenized_bbox
                        
                else: # not masked
                    masked_input_ids.append(token)
                    part_bbox.append(token_bbox)
                    if previous_bbox is not None:
                        targets += f" <loc_{previous_bbox[0]}> <loc_{previous_bbox[1]}> <loc_{previous_bbox[2]}> <loc_{previous_bbox[3]}>"
                        previous_bbox = None
                is_previous_masked = is_masked
            
            if previous_bbox is not None:
                targets += f" <loc_{previous_bbox[0]}> <loc_{previous_bbox[1]}> <loc_{previous_bbox[2]}> <loc_{previous_bbox[3]}>"

            total_input_ids.extend(masked_input_ids)
            total_targets += targets
            total_bbox.extend(part_bbox)

        return total_input_ids, total_targets, total_bbox
    