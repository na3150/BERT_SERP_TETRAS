from flask import Flask, request, jsonify
import time
import logging

from bert_demo.predict_bert_demo import inference as bert_demo_inference
from bert_demo.predict_bert_demo import load_model as bert_demo_load

#from sts.predict_sts import inference as sts_inference
#from sts.predict_sts import load_model as sts_load

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False # Response 재정렬 방지

# 정상일 경우는 콘솔 로그를 남기지 않는다.
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)

@app.route('/')
def check_auth():
    headers = request.headers
    auth = headers.get("Authorization")    
    if auth == 'Bearer xptmxmfktjrmsidgkemzheldgkqslek': # 테스트라서그냥하드코딩합니다
        return True
    else:
        return False

@app.route("/check_server", methods=["GET"])
def home():
    if check_auth():
        return "Hello!"
    else:
        return return_unauthorized()

@app.route("/predict_category", methods=["POST"])
def predict_category(): # bert_demo
    if check_auth():
        data = request.json
        start = time.time()
        result = bert_demo_inference(data, request.headers.get('Host'), request.remote_addr, request.headers.get('from-sts'), bert_demo_model, bert_demo_tokenizer)  
        return return_response_bert_demo(result, start)
    else:
        return return_unauthorized()

#@app.route("/predict_solution", methods=["POST"])
#def predict_sts_solution():    
#    if check_auth():
#        data = request.json        
#        start = time.time()
#        result = sts_inference(data, request.headers.get('Host'), request.remote_addr, sts_model, sts_train_index, sts_train_data, sts_train_index_all, sts_train_data_all, sts_sep_token)
#        return return_response_sts_solution(result, start)
#    else:
#        return return_unauthorized() 
      
def return_response_bert_demo(result, start):
    #print('-result in flask: ',result)
    pred1 = result["pred1"]
    pred2 = result["pred2"]    
    result_manager_name1 = result["manager_name1"]
    result_manager_name2 = result["manager_name2"]
    softmax1 = []    
    for s in result["softmax1"]:        
        softmax1.append(round(s*100,1))        
    softmax2 = []
    for s in result["softmax2"]:
        softmax2.append(round(s*100,1))        

    predictions = []
    for i in range(len(pred1)):                
        category = []
        manager_name = []
        softmax = []
        category.append(pred1[i])
        category.append(pred2[i])
        manager_name.append(result_manager_name1[i])
        manager_name.append(result_manager_name2[i])
        softmax.append(softmax1[i])
        softmax.append(softmax2[i])
        pred_per_data = dict({'category':category, 'manager_name':manager_name, 'softmax':softmax}) 
        predictions.append(pred_per_data)

    response = dict()
    response["predictions"] = predictions
    response["time"] = str(time.time()-start)
    
    return jsonify(response) 

def return_unauthorized():        
    return jsonify({"message": "ERROR: Unauthorized"}), 401

def load_models():
    global bert_demo_model, bert_demo_tokenizer
#   global sts_model, sts_train_index, sts_train_data, sts_train_index_all, sts_train_data_all, sts_sep_token

    # True=CPU / False=GPU
    bert_demo_model, bert_demo_tokenizer = bert_demo_load(no_cuda=True) 
#   sts_model, sts_train_index, sts_train_data, sts_train_index_all, sts_train_data_all, sts_sep_token= sts_load(no_cuda=False)

 
if __name__ == '__main__':
    load_models() # 서빙 시작시에 미리 모델들을 로딩해놓는다.
    app.run(host='0.0.0.0', port=5004)
    