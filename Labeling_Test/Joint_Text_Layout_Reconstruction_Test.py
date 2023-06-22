# Randomized Labeling Test for Joint Text-Layout Reconstruction (sentence version)

# 원래는 좀 더 세세하게 word 단위로 라벨링하려 했는데, 레이아웃에 중점을 두면 
# 문장 단위로 라벨링하는 게 나을 것 같아서 문장 단위로 했습니다

# 완성된 Prompt는 Tokenizer에 넣어질 예정
# 이 코드 참고해서 Dataset 클래스에 Labeling 코드 추가하면 될 듯!

import os
import json
import random

# Select sample num
idx = input('Input sample number : ')

test_path = f'./data/processed_sample_{idx}.json'

# Load json
with open(test_path, 'r', encoding='utf8') as f:
    try:
        data = json.load(f)
    except:
        data = {}

print(f'Number of Sentence : {len(data["form"])}')

for sentence in data['form']:
    print(sentence['text'])

# Randomly pick sentence to label 
target_sentence_idx = random.randint(0, len(data['form'])-1)

print(f'Targeted Sentence : {data["form"][target_sentence_idx]["text"]}')

target_sentence = data['form'][target_sentence_idx]

layout_token = ''

for i in range(len(target_sentence['box'])):
    n = target_sentence['box'][i]
    # Layout Normalization
    if i % 2 == 0:
        layout_token += f'<loc_{int(n*500/1280)}> '
    else:
        layout_token += f'<loc_{int(n*500/720)}> '

layout_token = layout_token[:-1]
label = '<extra_id_0> '+ data['form'][target_sentence_idx]['text'] + ' ' + layout_token
prompt = 'Joint Text-Layout Reconstruction.'

for i in range(len(data['form'])):
    if i == target_sentence_idx:
        # UDOP 논문 5페이지 참조. 얘네가 Layout Modeling 할 때 sentinel token을
        # <text_0>으로 했는데, UdopTokenizer에는 그런 token이 없어서 일단 <extra_id_0>으로 대체했음!
        # Task를 구별하기 위해 각 Task마다 다른 id를 쓰도록 수정했습니다
        # Layout Modeling => <extra_l_id_0>
        # Visual Text Recognition => <extra_t_id_0>
        # Joint Text_Layout Reconstruction => <extra_id_0>
        prompt += ' <extra_id_0>'
    else:
        prompt += ' ' + data['form'][i]['text']

print(f'\nPrompt : "{prompt}"\n')

print(f'Label : {label}')





