similar_keywords = [
    ['발표', '프레젠테이션', 'ppt', 'PPT', '피피티', '파워포인트', '발표용', 'PPT템플릿', '발표서식', '프리젠테이션', '프레젠테이션서식'],
    ['썸네일', '유튜브썸네일', '유튜브 썸네일', '섬네일', '유튜브섬네일', '유튜브'],
    ['이력서', '이력서 양식', '이력서 서식', '이력서양식', '이력서서식']
]

mapping = {keyword: group[0] for group in similar_keywords for keyword in group}

# 비슷한 키워드를 하나로 통일
def process_keyword(keywords):
    replaced_keywords = [mapping.get(keyword, keyword) for keyword in keywords]
    return list(dict.fromkeys(replaced_keywords))

if __name__ == '__main__':
  keywords = ['유튜브', '유튜브썸네일', '음악']
  print(process_keyword(keywords))