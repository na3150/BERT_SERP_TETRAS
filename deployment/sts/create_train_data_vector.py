import torch
import os
import csv

from tqdm import tqdm
from siamese_model.siamese_transformers import SiamesTransformers
from siamese_model.transformer import Transformer
from siamese_model.Pooling import Pooling
from siamese_model.CosineSimilarityLoss import CosineSimilarityLoss

# GPU 1번 사용
gpu_id = "cuda:0"
#gpu_id = "cpu"

FORCE_VECTORIZE = False
#FORCE_VECTORIZE = True # 캐쉬된 학습 벡터 데이터가 있어도 강제 재생성

data_dir = "./train_data"
model_dir = "./model"

## Data loading
def get_examples(target_file, skip_first=False):
    #   0       1        2      3       4        5      6     7
    # 모듈 / SR_ID_1 / 제목 / 본문 / SR_ID_2 / 제목 / 본문 / 점수  
    #examples = {'module': [], 'sr_id': [], 'title': [], 'voc': [], 'sr_id_2': [], 'sim_point': []}
    examples = []
    with open(target_file, encoding='utf-8 sig') as f:
        data = csv.reader(f, delimiter='\t')
        for id, row in enumerate(data):
            if skip_first and id == 0: continue            
            examples.append([row[0],row[1],row[2],row[3]])
            
    return examples
    
def generate_train_vector():    
    print('# Creating unique train file...')
    
    # 학습 원본 읽기
    train_examples_dup = get_examples(os.path.join(data_dir, "train.csv"))
    print("- train count(full) : ", len(train_examples_dup))
    
    # Train 중복 제거 : 학습 데이터에는 동일한 SR이 중복되어있음
    *train_examples, = map(list,{*map(tuple,train_examples_dup)})
    
    # 중복 제거한 학습 데이터를 파일로 저장
    with open(train_unique_file, 'w', encoding='utf-8 sig') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(train_examples)

    print("- train count(unique) : ", len(train_examples))
        
    ## extract vector    
    # 위에서 train을 벡터화환 결과를 아래 파일에 저장.
    train_vectors = []
 
    print("# Creating train vectors...")    
    sep_token = base_module.tokenizer.sep_token        
    for module, sr, title, voc in tqdm(train_examples):        
        train_vectors.append(model.encoder(title + ' ' + sep_token + ' ' + voc))
    
    # 결과 저장
    torch.save(train_vectors, train_vector_file_path)
    print("- train vector count : ", len(train_vectors))
    
    print('### Creating Finished to ', data_dir)    


## Siamese model loading
base_module = Transformer(model_dir, max_seq_length=512)
additional_module = [Pooling(base_module.get_word_embedding_dimension()),]
loss_module = CosineSimilarityLoss()
model = SiamesTransformers(base_module, additional_module, loss_module, device=gpu_id, fix_length=False)

# 중복 제거된 학습 파일과 벡터 파일이 모두 있으면, 강제 재생성일 때만 생성
# 둘은 행번호가 동일해야하므로 둘중 하나라도 없으면 모두 새로 생성
train_unique_file = data_dir + '/train_unique.csv'
train_vector_file_path = data_dir + '/train_vectors_base.bin'
if FORCE_VECTORIZE == True or (os.path.isfile(train_unique_file) == False and os.path.isfile(train_vector_file_path) == False):
    generate_train_vector()
else:
    print('# Found unique train file or train vector file')

