import argparse
import glob
import json
import logging
import os
import random
import copy
import csv
import requests
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification

##S-ERP 전처리기
import SerpPreProcessing
import easydict # argparse 대용

MODEL_DIR = 'bert_demo'

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

class SerpProcessor(DataProcessor):
    """Processor for the serp data set (GLUE version)."""
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
    
    def get_labels(self):
        """See base class."""
        list_f = open(MODEL_DIR+"/label.txt", "r", encoding="utf-8-sig")
        labels = []
        for l in list_f.readlines():
            labels.append(l[:-1])
        return labels

processors = {"serp" : SerpProcessor}
output_modes = {"serp" : "classification"}

def convert_examples_to_features_predict(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)        
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=0
            )
        )

    return features


# # Predict
def predict(request, model, tokenizer, prefix=""):
    preds_lists = []
    preds_lists2 = []
    logits_lists = []
    logits_lists2 = []

    eval_output_dir = args.output_dir
    eval_task = args.task_name
    mode = "predict"

    eval_dataset = prepare_predict_input(request, eval_task, tokenizer, mode)
    eval_dataloader = DataLoader(eval_dataset)
    
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    # Predict    
    preds = None                 
    for step, batch in enumerate(eval_dataloader):
        model.eval()        
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  
            ###### 이게 predict            
            outputs = model(**inputs)                        
            tmp_eval_loss, logits = outputs[:2]       
                                      
        # softmax
        for sm in torch.nn.functional.softmax(logits, dim=1).to("cpu").numpy():
            logits_lists.append(round(sm[np.argmax(sm)],3))
            # for second highest
            sm[np.argmax(sm)] = 0
            preds_lists2.append(np.argmax(sm))
            logits_lists2.append(round(sm[np.argmax(sm)],3))
                
        if preds is None:
            preds = logits.detach().cpu().numpy()
                
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)                
            
    preds = np.argmax(preds, axis=1)                
    preds_lists.extend(preds)   

    # 상세 결과 return (1/2순위 label 및 softmax)    
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    preds_lists_label = []
    preds_lists2_label = []
    for i in range(len(preds_lists)):
        preds_lists_label.append(label_list[preds_lists[i]])
        preds_lists2_label.append(label_list[preds_lists2[i]])    
     
    return preds_lists_label,preds_lists2_label,logits_lists,logits_lists2


# # 입력 데이터 처리
def prepare_predict_input(request, task, tokenizer, mode): # mode is train, dev(eval), valid
    processor = processors[task]()
    output_mode = output_modes[task]    
    label_list = processor.get_labels()
    examples = []
    i = 0

    # 입력 데이터 처리     
    lines = pre.convert_predict_req_json_to_list(request)
    for line in lines:
        valid, label, text_a = pre.pre_process_module_cls(line)

        if valid:
            i = i + 1
            guid = "%s-%s" % ('predict', i)            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=''))
                
    features = convert_examples_to_features_predict(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset

def set_args(no_cuda):    
    global args
    args = easydict.EasyDict({ 
        "output_dir" : MODEL_DIR+"/model",
        "model_type":"albert",
        "task_name" : "serp",
        "local_rank" : -1,
        "no_cuda" : no_cuda,
        "max_seq_length" : 512,
        
        "pre_conv_label" : False,#
        "pre_sample_train_max" : -1,#
        "pre_sample_train_min" : -1,#
        "pre_sample_test" : -1,#
        "pre_sample_by" : "LATEST", # "VOC_LENGTH"#
        "pre_delete_html" : True,
        "pre_to_lower" : True,
        "pre_delete_spc_chr" : True,               
        "pre_delete_stopword" : True,        
        "pre_conv_num_to_zero" : True,                
        "pre_convert_word" : True
        })    

def load_model(no_cuda=True):
    set_args(no_cuda)
    MODEL_CLASSES = {"albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)}

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()

    ##S-ERP - Start
    # SERP 전처리기 생성(global)
    global pre
    pre = SerpPreProcessing.SerpPreProcessing(args)

    ##S-ERP - End

    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    
    # fine-tuning한 모델을 다시 읽는 곳
    # Load a trained model and vocabulary that you have fine-tuned
    args.model_type = args.model_type.lower()    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=False, cls_token='<cls>', sep_token='<sep>')
    model.to(args.device)

    return model, tokenizer

def inference(request, host, sender_ip, from_sts, model, tokenizer):

    # Prediction
    pred1,pred2,softmax1,softmax2 = predict(request, model, tokenizer)

    # 서브 Label이 있는 모듈이면 서브 label 예측 실행
    # submod1, submod2, subsm1, subsm2 = predict_sub_modules(request,host,pred1,pred2,from_sts)
    # print('submod1=',submod1,'/','submod2=',submod2,'/', 'subsm1=',subsm1,'/','subsm2=',subsm2)

    # 연계된 시스템에서 담당자를 읽어온다.    
    manager_name1, manager_name2 = get_manager_from_if(pred1,pred2)

    response = dict({'pred1':pred1, 'pred2':pred2, 'manager_name1':manager_name1, 'manager_name2':manager_name2, 'softmax1':softmax1, 'softmax2':softmax2})    
    
    return response

def get_manager_from_if(pred1,pred2):
    # 결과 예시 : 3건을 동시에 예측하는 경우, 건별로 각 예측 결과의 1/2순위를 아래처럼 만들어주는 과정임
    #'pred1': ['FI', 'CO', 'CO'],
    #'pred2': ['BP', 'FI', 'FI'],
    #'softmax1': [1.0, 1.0, 1.0],
    #'softmax2': [0.0, 0.0, 0.0]
    #'manager_name1': ['김씨', '이씨', '박씨' ]
    #'manager_name2': ['최씨', '차씨', '홍씨'],
    
    manager_name1 = []
    manager_name2 = []

    # 여러건 예측일 경우 한건씩 처리.
    for i in range(len(pred1)):
        # 원래는 복잡한 비즈니스 로직이 들어가지만, 데모이므로 그냥 하드코딩. 실제 소스는 아래 주석 처리된 함수 참고바람
        manager_name1.append('이씨')
        manager_name2.append('김씨')

    return manager_name1, manager_name2
    
#def predict_sub_modules(request,host,pred1,pred2,from_sts):
    # 초기화
#    submod1 = len(pred1) * ['']
#    submod2 = len(pred1) * ['']
#    subsm1 = len(pred1) * [''] # 서브모듈 확률
#    subsm2 = len(pred1) * ['']

    # STS에서 호출했을 때는 서브모듈 예측 불필요
#    if from_sts == 'True': 
#        return submod1, submod2, subsm1, subsm2
        
#    # 서브 모듈 추가 예측 대상이면 API 호출
#    for i in range(len(pred1)):
#        # 1순위
#        if pred1[i] in PREDICT_SUB_MODULE:                        
#            submod1[i], subsm1[i] = get_sub_module_from_api_call(request,host,pred1[i],i)
            #print('-PREDICT SUBMODULE : ',i, '/R1/', pred1[i],'/',submod1[i])
        # 2순위
#        if pred2[i] in PREDICT_SUB_MODULE:
#            submod2[i], subsm2[i] = get_sub_module_from_api_call(request,host,pred2[i],i)
            #print('-PREDICT SUBMODULE : ',i, '/R2/', pred2[i],'/',submod2[i])
        #print('\n')
           
#    return submod1, submod2, subsm1, subsm2

#def get_sub_module_from_api_call(request_src,host,module,i): # i=데이터번호
#    sub_module = ''
#    sub_softmax = 0

    # request에서 해당 영역 데이터를 추출하며 서브모듈용 request를 만든다.
#    datas = []
#    datas.append(dict(request_src["datas"][i]))
#    request = dict()
#    request["datas"] = datas

#    url = 'http://localhost:' + host.split(':')[1] + '/' + PREDICT_SUB_MODULE_URL[PREDICT_SUB_MODULE.index(module)]
    #print('$$$ URL = ', url)
#    headers = {'Content-Type': 'application/json', 'Authorization' : MODULE_PRED_AUTH_TOKEN}    
#    response = requests.post(url, headers=headers, data=json.dumps(request))

#    if response.status_code == 200:          
#        predictions = json.loads(response.text)["predictions"]
#        sub_module = predictions[0]["system"][0] # 1순위만 사용
#        sub_softmax = round(predictions[0]["softmax"][0]/100,3)
        
#    return sub_module, sub_softmax

#def get_service_manager_from_solman(pred1,pred2,submod1,submod2,softmax1,softmax2,subsm1,subsm2,request,from_sts):
    # 결과 에시 : 3건을 동시에 예측하는 경우, 건별로 각 예측 결과의 1/2순위를 아래처럼 만들어주는 과정임
    #'pred1': ['FI', 'CO', 'CO'],
    #'pred2': ['BP', 'FI', 'FI'],
    #'softmax1': [1.0, 1.0, 1.0],
    #'softmax2': [0.0, 0.0, 0.0]
    #'service1': ['S-ERP_FI', 'S-ERP_CO', 'S-ERP_CO' ]
    #'service2': ['S-ERP_BP', 'S-ERP_FI', 'S-ERP_FI'],

    # 솔맨에서 서비스명, 추천 담당자를 읽어온다. 
#    service1 = []
#    service2 = []
#    manager_name1 = []
#    manager_name2 = []
#    manager_mail1 = []
#    manager_mail2 = []
    
    # request에서 서비스콜ID, 고객사 코드와 제목을 추가로 추출해서 파라미터로 같이 넘겨야한다.
#    call_id = pre.get_field_values_from_request(request,'sr_id')
#    company_code = pre.get_field_values_from_request(request,'company_code')
#    title = pre.get_field_values_from_request(request,'title')
#    req_date = pre.get_field_values_from_request(request,'req_datetime')
    
    # 솔맨 Connection
#    conn = Connection(**solman_config._sections[SOLMAN_SERVER]) 

    # 여러건 예측일 경우 한건씩 RFC 호출해야한다.    
#    for i in range(len(pred1)):
        # 파라미터 생성 
#        t_predict = []
#        iv_from_sts = 'X' if from_sts == 'True' else ''            
        
        # 1순위
#        t_predict.append({'PRED_MODULE': pred1[i], 'PRED_SUBMODULE':submod1[i], 'SOFTMAX':str(softmax1[i]), 'SOFTMAX_SUB':str(subsm1[i]), 'SYSTEM': '', 'SERVICE_NAME': '', 'MANAGER_NAME': '', 'MANAGER_MAIL': ''})
        # 2순위
#        t_predict.append({'PRED_MODULE': pred2[i], 'PRED_SUBMODULE':submod2[i], 'SOFTMAX':str(softmax2[i]), 'SOFTMAX_SUB':str(subsm2[i]), 'SYSTEM': '', 'SERVICE_NAME': '', 'MANAGER_NAME': '', 'MANAGER_MAIL': ''})
        
#        result = conn.call("ZA0CMZ_GET_SR_SERVICE_MANAGER", IV_CALL_ID=call_id[i], IV_TITLE=title[i] , IV_COMPANY_CODE=company_code[i], IV_FROM_STS=iv_from_sts, IV_DATE=req_date[i][0:8], T_PREDICT=t_predict)
#        t_result = result['T_PREDICT']

        #print("- t_result from solman = ", t_result,'\n' )

        # 서비스/시스템/담당자/확률 등 추출, 
#        for rank, svc in enumerate(t_result):
#            if rank == 0: # 1순위
#                pred1[i] = svc['SYSTEM'] # 예측 모듈은 최종 시스템(구성 항목)으로 변경
#                submod1[i] = svc['PRED_SUBMODULE']
#                softmax1[i] = float(svc['SOFTMAX'])
#                service1.append(svc['SERVICE_NAME'])
#                manager_name1.append(svc['MANAGER_NAME'])
#                manager_mail1.append(svc['MANAGER_MAIL'])

#            else: # 2순위
#                pred2[i] = svc['SYSTEM'] # 예측 모듈은 최종 시스템(구성 항목)으로 변경
#                submod2[i] = svc['PRED_SUBMODULE']
#                softmax2[i] = float(svc['SOFTMAX'])
#                service2.append(svc['SERVICE_NAME'])
#                manager_name2.append(svc['MANAGER_NAME'])
#                manager_mail2.append(svc['MANAGER_MAIL'])
            #print('# ', str(i+1),'-',str(rank+1), ':', svc['PRED_MODULE'], '/', svc['SYSTEM'], '/', svc['PRED_SUBMODULE'], '/', svc['SOFTMAX'], '/', svc['SERVICE_NAME'], '/', svc['MANAGER_NAME'], '/', svc['MANAGER_MAIL'])

#    conn.close()

#    return pred1, pred2, service1, service2, submod1, submod2, manager_name1, manager_name2, manager_mail1, manager_mail2, softmax1, softmax2

#def predict_with_tcode(request,pred1,pred2,submod1,submod2,softmax1,softmax2,subsm1,subsm2):
    # TCODE가 있으면 RFC로 TCODE의 모듈/서브모듈을 얻는다.
#    datas = request["datas"]
#    for i,data in enumerate(datas):
#        tcodes = pre.get_tcodes_from_title_voc(data["title"] + ' ' + data["voc"])
#        if len(tcodes) > 0:                 
#            conn = Connection(**solman_config._sections[SOLMAN_SERVER]) # 솔맨 Connection
#            result = conn.call("ZA0CMZ_GET_TCODE_MODULE_SUBMOD", IV_TCODES=tcodes, IV_CUR_MODULE=pred1[i])
#            ev_module = result['EV_MODULE']
#            ev_submodule = result['EV_SUBMODULE']            
            #print('ev_module = ', ev_module, '/', 'ev_submodule = ',ev_submodule ,'/','tcodes = ',tcodes)
            # TCODE 예측 결과를 1순위로, 기존 1순위는 2순위로 내린다.
#            if ev_module != '' and ev_module != pred1[i]:
#                pred2[i] = pred1[i]
#                submod2[i] = submod1[i]
#                softmax2[i] = softmax1[i] - 0.001 # 100% 면 안 되므로...                
#                if subsm1[i] != '':
#                    subsm2[i] = str(float(subsm1[i]) - 0.001)                    
#                pred1[i] = ev_module
#                submod1[i] = ev_submodule
#                softmax1[i] = 1
#                if ev_submodule != '':
#                    subsm1[i] = 1
#                else:
#                    subsm1[i] = 0

#            conn.close()               

#    return pred1,pred2,submod1,submod2,softmax1,softmax2,subsm1,subsm2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       