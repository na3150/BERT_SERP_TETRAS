import os
import csv
import faiss
import torch
import numpy as np
import requests
import json
from pyrfc import Connection
from configparser import ConfigParser

from sts.siamese_model.siamese_transformers import SiamesTransformers
from sts.siamese_model.transformer import Transformer
from sts.siamese_model.Pooling import Pooling
from sts.siamese_model.CosineSimilarityLoss import CosineSimilarityLoss

##S-ERP 전처리기
import SerpPreProcessing
import easydict

MAX_RANK = 5 # Top 5
MIN_SIM_RATE = 85 # 유사도 임계치(%) -> 단, 최종 임계도는 솔맨에서 한번 더 거름
MODEL_DIR = 'sts/model'
TRAIN_DATA_DIR = 'sts/train_data/'
MODULE_PRED_URL = 'predict_svc_mgr'
MODULE_PRED_AUTH_TOKEN = 'Bearer xxxxxxxxxxxxxxxxxx'
SOLMAN_SERVER = "connection_prd"
###### Test ######
#SOLMAN_SERVER = "connection_dev"
#MIN_SIM_RATE = 50 # 유사도 임계치(%)
###### Test ######

def set_args(no_cuda):    
    global args
    args = easydict.EasyDict({         
        "model":"albert",
        "no_cuda" : no_cuda,
        "max_seq_length" : 512,
        
        "pre_conv_label" : True,
        "pre_sample_train_max" : -1,
        "pre_sample_train_min" : -1,
        "pre_sample_test" : -1,
        "pre_sample_by" : "LATEST",
        "pre_delete_html" : True,
        "pre_to_lower" : True,
        "pre_delete_spc_chr" : True,       
        "pre_delete_d_in_title" : True,
        "pre_delete_stopword" : True,
        "pre_delete_stopword_template" : True,
        "pre_conv_num_to_zero" : True,
        "pre_conv_name" : True,
        "pre_delete_srid" : False,
        "pre_convert_word" : True,
        "pre_tcode_to_title" : True,
        "pre_title_twice" : True  
        })  
        
# ## Siamese model loading
def load_model(no_cuda = True):
    print("####################### START OF LOADING STS MODEL ###########################")
        
    # SET cpu or GPU
    if no_cuda == True:
        gpu_id = "cpu"
    else:
        gpu_id = "cuda:0"
    
    # Load sts trained moodel
    base_module = Transformer(MODEL_DIR, max_seq_length=512)
    additional_module = [Pooling(base_module.get_word_embedding_dimension()),]
    loss_module = CosineSimilarityLoss()
    model = SiamesTransformers(base_module, additional_module, loss_module, device=gpu_id, fix_length=False)
    
    set_args(no_cuda)
    
    ##S-ERP - Start
    # SERP 전처리기 생성(global)
    global pre
    pre = SerpPreProcessing.SerpPreProcessing(args)

    # 솔면 연동 설정은 로딩시 미리 읽어놓고, Connection은 호출시마다 수행한다.
    global solman_config
    solman_config = ConfigParser()
    solman_config.read("saplib/pyrfc/sapnwrfc.cfg")
    ##S-ERP - End

    # Load unique train data then build index(faiss)
    train_index, train_datas, train_index_all, train_datas_all = load_and_build_train_index_data()
    if train_index == None or train_datas == None:
        return None, None, None, None, None

    print("####################### END OF LOADING STS MODEL ###########################")

    # Return loaded model and vectorized index(faiss-cosine sim.)
    return model, train_index, train_datas, train_index_all, train_datas_all, base_module.tokenizer.sep_token


# ## Data loading
def get_examples_from_unique(target_file):
    examples = []    
    with open(target_file, encoding='utf-8-sig') as f:
        data = csv.reader(f, delimiter='\t')        
        for row in data:
            examples.append(row)
    
    return examples

def load_and_build_train_index_data():
    
    # CPU로 벡터 변환은 정말 오래 걸리므로, 미리 변환된 파일을 읽기만 한다.
    # 이때, 변환된 벡터와 중복 제거된 학습 데이터는 순서가 일치해야하므로 동시에 존재해야만 한다.
    train_vectors = []
    train_datas = []
    train_vector_file_path = TRAIN_DATA_DIR + 'train_vectors_base.bin'
    train_unique_file = TRAIN_DATA_DIR + 'train_unique.csv'    

    if os.path.isfile(train_unique_file) and os.path.isfile(train_vector_file_path):
        print("- Found vector & unique train file : ", train_vector_file_path, '/',train_unique_file )
        train_vectors = torch.load(train_vector_file_path)
        train_datas = get_examples_from_unique(train_unique_file) 
        print("- Length of unique train : ",len(train_datas))
        print("- Length of vector : ",len(train_vectors))
        if len(train_vectors) == 0 or len(train_datas) == 0:
            print("##### ERROR : Blank vector or unique train file")
            return None, None, None, None
        elif len(train_vectors) != len(train_datas):
            print("##### ERROR : Different length of vector and unique train file")
            return None, None , None, None     
            
    else:    
        print("##### ERROR : Could not find vector or unique train file")
        return None, None, None, None
    
    # train data와 train vector를 label(모듈)별로 분리
    train_datas_by_label = {}
    train_vectors_by_label = {}
    label_list = []
    for i, example  in enumerate(train_datas):
        module = example[0]
        sr = example[1]
        title = example[2]
        voc = example[3]
        # 새로운 label이면 dict 생성
        if module not in label_list: # new label
            label_list.append(module)
            train_datas_by_label[module] = []
            train_vectors_by_label[module] = []
        
        # 추가
        train_datas_by_label[module].append([module, sr, title, voc])
        train_vectors_by_label[module].append(train_vectors[i])
    
    # faiss index 생성(코사인 유사도)        
    print("- Start building sts train index...")    
    train_index_by_label = {}
    for module in train_vectors_by_label.keys():
        train_vec_array = np.asarray(train_vectors_by_label[module]).squeeze(axis=1)        
        train_index_by_label[module] = faiss.IndexFlatIP(train_vec_array.shape[1])        
        faiss.normalize_L2(train_vec_array)    
        train_index_by_label[module].add(train_vec_array)
        print(module, ':',type(train_vec_array), train_vec_array.shape)
    
    # 모듈 구분 없는 인덱스도 생성(코사인 유사도)
    train_vec_array_all = np.asarray(train_vectors).squeeze(axis=1)   
    train_index_all = faiss.IndexFlatIP(train_vec_array_all.shape[1])        
    faiss.normalize_L2(train_vec_array_all)    
    train_index_all.add(train_vec_array_all)
        
    return train_index_by_label, train_datas_by_label, train_index_all, train_datas


def inference(request, host, sender_ip, model, train_index_by_label, train_datas_by_label, train_index_all, train_datas_all, sep_token):    
    # json에서 데이터 추출 후 전처리
    predict_examples = pre.convert_predict_req_json_to_list_for_sts(request)

    # 모듈 먼저 예측 : 모듈 예측과 유사도 예측 req 포맷이 동일하므로, 그대로 모듈 예측 API 실행하면 된다.
    module_list = get_module_from_api_call(request, host)
    #print('■ module_list =', module_list)

    # 예측 데이터 벡터 변환. CPU로 건당 0.5초 소요
    predict_vectors = []
    predict_examples_valid = []
    for i, example in enumerate(predict_examples):
        example[0] = module_list[i] # 예측된 모듈
        if example[0] == '' or example[0]  not in train_datas_by_label.keys(): # 학습 데이터가 없는 모듈이면 제외
            continue

        # 전처리
        #print("전처리 전 - ", example[0], '/', example[2], '/', example[3])
        valid, module, title_voc, title_voc2 = pre.pre_process_sts(example, sep_token)        
        if valid == True:
            example[0] = module # module이 바뀔 수도 있음. 거의 없겠지만...
            predict_examples_valid.append(example)
            #print("- ", example[1], ':', example[0], '/', title_voc)
            predict_vectors.append(model.encoder(title_voc))    
            
    predict_vec_array = np.asarray(predict_vectors)

    # RFC 실행시 필요한 추가 데이터들
    company_code = pre.get_field_values_from_request(request,'company_code')
    title = pre.get_field_values_from_request(request,'title')
    sr_id = pre.get_field_values_from_request(request,'sr_id')
    
    ## 코사인 유사도로 찾기   
    sr_id_sim_all = []
    srvc_req_key_sim_all = []
    service_sim_all = []
    solution_sim_all = []    
    softmax_sim_all = [] 
    for query_id, example in enumerate(predict_examples_valid):     
        module = example[0]   
        faiss.normalize_L2(predict_vec_array[query_id])        
        D, I = train_index_by_label[module].search(predict_vec_array[query_id], MAX_RANK*5) # 해결책 품질을 추가 고려해야하므로 여유있게 5배로 찾아온다.
        rank = I.squeeze(axis=0)    
        distance = D.squeeze(axis=0)   
                        
        # 찾은 유사 SR
        work_id_sim = [] # 학습 데이터에는 서비스콜이 아닌 작업 요청 ID가 있음        
        softmax_src = []
        for i, r in enumerate(rank):     
            # 코사인 유사도를 확률로 변환 : 원래 -1 ~ 1 이지만, 0 이하는 사실상 없으므로 무시하고 0~1 사이를 확률로 치환한다.
            if distance[i] < 0:
                sim_rate = 0
            else:
                sim_rate = int(distance[i] * 100)

            # 유사도 임계값 이상일 때만 사용
            if sim_rate >= MIN_SIM_RATE:                
                work_id_sim.append(train_datas_by_label[module][r][1])                            
                softmax_src.append(sim_rate)                        
                #print('-OK  :', train_datas_by_label[module][r][0],'/',str(sim_rate),'/',train_datas_by_label[module][r][1],'/',train_datas_by_label[module][r][2])
            #else:
                #print('-BAD :', train_datas_by_label[module][r][0],'/',str(sim_rate),'/',train_datas_by_label[module][r][1],'/',train_datas_by_label[module][r][2])

        # 솔맨에서 SR 해결책 등 추가 정보를 읽어온다.
        sr_id_sim, srvc_req_key_sim, service_sim, solution_sim, softmax_sim  = get_sr_info_from_solman(work_id_sim, softmax_src, sr_id[query_id], company_code[query_id])
        
        # 같은 모듈 안에서 임계치 이상을 1건도 못 찾은 경우는 전체 모듈에서 찾아본다.
        if len(sr_id_sim) == 0:
            work_id_sim = []
            softmax_src = []
            D_all, I_all = train_index_all.search(predict_vec_array[query_id], MAX_RANK*5)
            rank_all = I_all.squeeze(axis=0)    
            distance_all = D_all.squeeze(axis=0)   
            
            for i, r in enumerate(rank_all):                
                if module == train_datas_all[r][0]: # 다른 모듈일 때만
                    continue
                    
                if distance_all[i] < 0:
                    sim_rate = 0
                else:
                    sim_rate = int(distance_all[i] * 100)                    
                
                # 유사도 임계값 이상일 때만 사용
                if sim_rate >= MIN_SIM_RATE:                
                    work_id_sim.append(train_datas_all[r][1])                            
                    softmax_src.append(sim_rate)                        
                    #print('- Found in Other module :', module, '-', train_datas_all[r][0], '/', sr_id[query_id], '-', train_datas_all[r][1])
                    
            # 솔맨에서 SR 해결책 등 추가 정보를 읽어온다.
            sr_id_sim, srvc_req_key_sim, service_sim, solution_sim, softmax_sim  = get_sr_info_from_solman(work_id_sim, softmax_src, sr_id[query_id], company_code[query_id])
        
        sr_id_sim_all.append(sr_id_sim)
        srvc_req_key_sim_all.append(srvc_req_key_sim)
        service_sim_all.append(service_sim)
        solution_sim_all.append(solution_sim)
        softmax_sim_all.append(softmax_sim)

        # ITSM에서 호출시는 솔맨에 Log를 남긴다.        
        if sender_ip in ['xx.xx.xxx.xxx','xx.xx.xx.xxx']:
            save_predict_log_to_solman(sr_id[query_id], title[query_id], company_code[query_id], sr_id_sim, service_sim, solution_sim, softmax_sim, module)

    # 한건도 없으면 예외 처리
    if len(sr_id_sim_all) == 0:
        sr_id_sim_all.append([])
        srvc_req_key_sim_all.append([])
        service_sim_all.append([])
        solution_sim_all.append([])
        softmax_sim_all.append([])
        
    response = dict({'sr_id':sr_id_sim_all, 'srvc_req_key':srvc_req_key_sim_all, 'service':service_sim_all, 'solution':solution_sim_all, 'softmax':softmax_sim_all})    
    #print('response=',response)
    return response

def get_module_from_api_call(request, host):
    # 모듈 분류 API를 호출하여 모듈을 예측한다.
    module_list = []
    # STS에서 모듈 API호출시는 헤더에 from-sts를 넘겨서 담당자 얻기 위한 솔맨 RFC 호출은 하지 않도록 한다.
    headers = {'Content-Type': 'application/json', 'Authorization' : MODULE_PRED_AUTH_TOKEN, 'from-sts' : 'True'}
    url = 'http://localhost:' + host.split(':')[1] + '/' + MODULE_PRED_URL
    response = requests.post(url, headers=headers, data=json.dumps(request))
    
    #print("response.status_code-",response.status_code)
    if response.status_code == 200:  
        predictions = json.loads(response.text)["predictions"]        
        #print("predictions-",predictions)
        for pred in predictions:                  
            module_list.append( pred["system"][0] ) # 1/2순위가 있으므로 1순위만 사용  

    return module_list    

def get_sr_info_from_solman(work_id_src, softmax_src, sr_id_src, company_code):
    sr_id = []
    srvc_req_key = []
    service = []
    solution = []
    softmax = []    

    # 유사 SR이 있을 때만
    if len(work_id_src) > 0:
        # 파라미터 생성
        t_sr = []
        for i, work_id in enumerate(work_id_src):
            t_sr.append({'WORK_ID': work_id, 'CALL_ID': '', 'SRVC_REQ_KEY': '', 'SOFTMAX':str(softmax_src[i]), 'SERVICE_NAME': '', 'PROC_DESC': ''})

        # 솔맨 Connection 후 RFC 호출
        conn = Connection(**solman_config._sections[SOLMAN_SERVER]) 
        result = conn.call("ZA0CMZ_GET_SR_SOLUTION", T_SR=t_sr, IV_MAX_RANK=str(MAX_RANK), IV_COMPANY_CODE=company_code, IV_CALL_ID=sr_id_src )
        t_sr_result = result['T_SR']

        # SR 상세 데이터 반환        
        for sr in t_sr_result:
            sr_id.append(sr['CALL_ID'])
            srvc_req_key.append(sr['SRVC_REQ_KEY'])
            service.append(sr['SERVICE_NAME'])
            solution.append(pre.make_sts_solution_beautiful(sr['PROC_DESC'])) # 템플릿 등 제거후 보여준다.
            softmax.append(softmax_src[work_id_src.index(sr['WORK_ID'])]) # 채택된 SR의 확률
            #print('------------------------------------------\n', pre.make_sts_solution_beautiful(sr['PROC_DESC']))        
        
        conn.close()
    
    return sr_id, srvc_req_key, service, solution, softmax

def save_predict_log_to_solman(sr_id_src, title, company_code, sr_id_sim, service_sim, solution_sim, softmax_sim, module):

    # 파라미터 생성
    t_result = []
    for i, sr_id in enumerate(sr_id_sim):
        t_result.append({'RANK':str(i), 'CALL_ID_SIM': sr_id, 'SOFTMAX':str(softmax_sim[i]), 'SERVICE_NAME':service_sim[i], 'CI_NAME':module, 'CONTENTS':solution_sim[i]})

    # 솔맨 Connection 후 RFC 호출
    conn = Connection(**solman_config._sections[SOLMAN_SERVER]) 
    result = conn.call("ZA0CMZ_LOG_STS_PREDICT_RESULT", T_RESULT=t_result, IV_CALL_ID=sr_id_src, IV_COMPANY_CODE=company_code, IV_TITLE=title)    
    conn.close()
    
