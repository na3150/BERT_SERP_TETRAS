import re
from bs4 import BeautifulSoup
import pandas as pd
    
class SerpPreProcessing:
    def __init__(self, args):        
        
        # 분류용 필드 순서 : 입력 데이터 컬럼수와 컬럼 POS를 반드시 지정해야함
        self.TOTAL_COL = 4 # (=4개)
        self.LABEL_POS = 0        
        self.ID_POS  = 1
        self.TITLE_POS = 2
        self.TEXT_POS = 3        
        
        self.args = args
        self.label_conversion = {}
        self.stopwords = []
        self.stopwords_sub = '' # |로 붙여놓아서 re.sub에서 사용        
        self.sampled_label = [] # 샘플링된 Label        
        self.mean_text_length = 0 # 학습된 TEXT의 평균 길이        
        
        self.word_conversion_regex = [] # 유의어 regex list(size 제한이 있어서 list화)
        self.word_conversion_dict = {} # 유의어 regex dictionary

        self.test_data = [] # 상세 테스트 결과 출력용
        self.test_data_srid = [] # eval 여러번 실행되면 srid 중복돼서...
        
        # # 삭제/변환 대상 Label 읽기
        # if self.args.pre_conv_label == True:
            # self.__read_label_conversion_delete_file()
        
         # 불용어 파일 읽기
         if self.args.pre_delete_stopword == True:
             self.__read_stopwords_file()
             #print("- Stopwords :", self.stopwords)
    
        # # 유의어 변경 매핑 읽기
        # if self.args.pre_convert_word == True:
            # self.__read_word_conversion_file()  

    def __read_label_conversion_delete_file(self):        
        mapping_f = open("label_conversion_delete.txt", "r", encoding="utf-8-sig")
        for m in mapping_f.readlines():            
            line = m.split('\t')
            self.label_conversion[line[0]] = line[1][:-1]

    def __read_stopwords_file(self):        
        to_esc = ':"|()[]<>*'
        # 공용 불용어
        stopwords_f = open("stopwords.txt", "r", encoding="utf-8-sig")        
        for w in stopwords_f.readlines():
            self.stopwords.append(w[:-1])
            for c in to_esc:
                w = w.replace(c, '\\'+c)
            if self.stopwords_sub == '':
                self.stopwords_sub = w[:-1]
            else:
                self.stopwords_sub = self.stopwords_sub + '|' + w[:-1]                

                     
    def __read_word_conversion_file(self): 
        conv_f = open("word_conversion.txt", "r", encoding="utf-8-sig")
        
        word_conversion = {}
        i = 0
        for c in conv_f.readlines():            
            i = i + 1
            line = c.split('\t')
            self.word_conversion_dict[line[0]] = line[1][:-1]
            word_conversion[line[0]] = line[1][:-1]
            if i == 10:
                self.word_conversion_regex.append(re.compile("(%s)" % "|".join(map(re.escape, word_conversion.keys()))))
                word_conversion = {}
                i = 0
        if i > 0:
            self.word_conversion_regex.append(re.compile("(%s)" % "|".join(map(re.escape, word_conversion.keys()))))
            
            
    def get_converted_label(self, label):
        converted_label = label
        if self.args.pre_conv_label == True and label != '':
            if label in self.label_conversion:
                if self.label_conversion[label] == 'delete':
                    converted_label = ''
                else:
                    converted_label = self.label_conversion[label]
        return converted_label
                    
    # 분류용 전처리
    def pre_process_module_cls(self, line):        
        label = line[self.LABEL_POS]                        
        text = line[self.TEXT_POS]                
        title = line[self.TITLE_POS]                
                
        # Label 변환or삭제        
        if self.args.pre_conv_label == True and label != '':            
            if label in self.label_conversion:
                if self.label_conversion[label] == 'delete':
                    return False, '', ''            
                else:
                    label = self.label_conversion[label]
                     
        # HTML 태그 제거
        if self.args.pre_delete_html == True:            
            title = self.__pre_delete_html(title)
            text = self.__pre_delete_html(text)
        
        # 불용어 제거 : pre_delete_stopword
        if self.args.pre_delete_stopword == True:
            title = self.__pre_delete_stopword(title)
            text = self.__pre_delete_stopword(text)
            
        # 소문자 변환
        if self.args.pre_to_lower == True:
            title = self.__pre_to_lower(title)
            text = self.__pre_to_lower(text)

        # 유의어 변환
        if self.args.pre_convert_word == True:
            title = self.__pre_convert_word(title)
            text = self.__pre_convert_word(text)

        # 특수문자 제거(단,문장 구분 부호(.?!;)은 유지)
        if self.args.pre_delete_spc_chr == True:
            title = self.__pre_delete_spc_chr(title)
            text = self.__pre_delete_spc_chr(text)
       
        # 숫자는 0으로 변경
        if self.args.pre_conv_num_to_zero == True:            
            title = self.__pre_conv_num_to_zero(title)
            text = self.__pre_conv_num_to_zero(text)
                        
        # 최종 정돈 및 null 제거 후 BERT 토큰 합치기
        title = self.__pre_complete_clean(title)
        text = self.__pre_complete_clean(text)
        if text != '' and title != '':
            text_a = '<cls>' + title + '<sep>' + text + '<sep>'
            return True, label, text_a
        else:
            return False, '', '' 
    
    
    def __pre_delete_html(self,content):
        return BeautifulSoup(content, "html.parser").get_text()
            
        
    def __pre_to_lower(self,content):
        return content.lower()
    
        
    
    def __pre_delete_spc_chr(self,content):
        return re.sub("[-_=+,#/\:^$@*\"※♥■~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]", " ", content)
        

    def __pre_conv_num_to_zero(self,content):
        return re.sub(r"\d+", "0", content)
    
    
    def __pre_delete_stopword(self,content):
        return re.sub(self.stopwords_sub, " ", content)
        
    def __pre_convert_word(self,content):
        after = content
        for regex in self.word_conversion_regex:
            after = regex.sub(lambda mo: self.word_conversion_dict[mo.string[mo.start():mo.end()]], after)
 
        return after
            
    def __pre_complete_clean(self,content):
        return content.strip()
    
    
    def sampling_train(self,lines):
        raw_data = []        
        for line in lines:
            line = line[:-1].split('\t')
            raw_data.append(line)
            
        if self.args.pre_sample_by == 'TEXT_LENGTH':
            raw_data_df = self.__sampling_train_by_text_length(raw_data)        
        elif self.args.pre_sample_by == 'LATEST':
            raw_data_df = self.__sampling_train_by_latest(raw_data)
        else:
            return lines
        
        # pre_sample_train_min 이상인 label만 pre_sample_train_max까지만 샘플링   
        label_count = raw_data_df[self.LABEL_POS].value_counts()[raw_data_df[self.LABEL_POS].value_counts() >= self.args.pre_sample_train_min]
        self.sampled_label = label_count.keys().tolist()
        print("# Total train count per label : \n", label_count)        
                            
        return self.__finalize_sampling(raw_data_df, self.args.pre_sample_train_max)
        
        
    def sampling_test(self,lines):
        raw_data = []        
        for line in lines:
            line = line[:-1].split('\t')
            raw_data.append(line)
            
        if self.args.pre_sample_by == 'TEXT_LENGTH':
            raw_data_df =  self.__sampling_test_by_text_length(raw_data)        
        elif self.args.pre_sample_by == 'LATEST':
            raw_data_df =  self.__sampling_test_by_latest(raw_data)
        else:
            return lines
        
        # 학습시 샘플링된 label만 최대 pre_sample_test 만큼 (학습된 label이면 갯수가 적어도 test에 사용)
        label_count = raw_data_df[self.LABEL_POS].value_counts()        
        print("# Total test count per label(before sampling) : \n", label_count)        
        
        return self.__finalize_sampling(raw_data_df, self.args.pre_sample_test)
        
        
    def __sampling_train_by_latest(self,lines):    
        print("### START of TRAIN sampling by LATEST. Samples per label is ", self.args.pre_sample_train_min)
        
        # Label별 최신 ID 순으로 정렬 및 카운팅
        raw_data_df = pd.DataFrame.from_records(lines)
        raw_data_df.sort_values([self.LABEL_POS, self.ID_POS], ascending=[True, False], inplace=True)
        return raw_data_df

    
    def __sampling_test_by_latest(self,lines):     
        print("### START of TEST sampling by LATEST. Sampled labels are ", self.sampled_label)
        
        # Label별 최신 ID 순으로 정렬 및 카운팅
        raw_data_df = pd.DataFrame.from_records(lines)
        raw_data_df.sort_values([self.LABEL_POS, self.ID_POS], ascending=[True, False], inplace=True)
        return raw_data_df
    
    
    def __sampling_train_by_text_length(self,lines):    
        print("### START of TRAIN sampling by TEXT LENGTH. Samples per label  is ", self.args.pre_sample_train_min)
        
        # text 평균 길이 및 평균길이 오차 구하기 순으로 정렬/카운팅
        raw_data_df = pd.DataFrame.from_records(lines)
        raw_data_df['length'] = raw_data_df[self.TEXT_POS].apply(lambda x : len(x))
        self.mean_text_length = int(raw_data_df['length'].mean())
        raw_data_df['mean_dif'] = raw_data_df['length'].apply(lambda x : abs(self.mean_text_length-x))
        raw_data_df.sort_values([self.LABEL_POS, 'mean_dif'], ascending=[True, True], inplace=True)
        return raw_data_df
        
    
    def __sampling_test_by_text_length(self,lines):
        print("### START of TEST sampling by TEXT LENGTH. Sampled labels are ", self.sampled_label)
        
        # 학습된 text의 평균길이(self.mean_text_length) 기준 오차 순으로 정렬/카운팅
        raw_data_df = pd.DataFrame.from_records(lines)
        raw_data_df['length'] = raw_data_df[self.TEXT_POS].apply(lambda x : len(x))        
        raw_data_df['mean_dif'] = raw_data_df['length'].apply(lambda x : abs(self.mean_text_length-x))
        raw_data_df.sort_values([self.LABEL_POS, 'mean_dif'], ascending=[True, True], inplace=True)
        return raw_data_df
    
        
    def __finalize_sampling(self,raw_data_df,max_sample):
        sorted = raw_data_df.values.tolist()        
        pre_label = ''
        cur_count = 0
        lines_return = []
        for line in sorted:
            label = line[self.LABEL_POS]
            if label != pre_label:
                cur_count = 1
            else:
                cur_count = cur_count + 1
                
            if label in self.sampled_label and cur_count <= max_sample:                                
                lines_return.append(line[:self.TOTAL_COL])                
                            
            pre_label = label
                            
        return lines_return     
    
    
    def record_test_data(self, line, text):
        # LABEL_POS = 원본 label
        if line[self.ID_POS] not in self.test_data_srid:
            record = [line[self.LABEL_POS], line[self.ID_POS], text ]
            self.test_data.append(record)
            self.test_data_srid.append(line[self.ID_POS])
    
    ## Predict API는 JSON으로 넘어오므로 list 형태로 컨버전(분류용)
    def convert_predict_req_json_to_list(self, req):
        req_as_list = []    
        datas = req["datas"]
        for data in datas:
            line = self.TOTAL_COL * [None] 
            line[self.ID_POS] = data["id"]            
            line[self.TITLE_POS] = data["title"]                        
            line[self.TEXT_POS] = data["text"]            
            line = ["" if v is None else v for v in line]
            req_as_list.append(line)
            
        return req_as_list
    
    # 요청에서 원하는 필드만 추출
    def get_field_values_from_request(self,req,fieldname):
        values = []    
        datas = req["datas"]
        for data in datas:
            values.append(data[fieldname])
        return values

    # TCODE 추출
    def get_tcodes_from_title_text(self,content):        
        tcode_pattern = re.compile('Z......\d\d\d\d')                
        num_list = ['0','1','2','3','4','5','6','7','8','9']
        tokens = content.split(' ')        
        tcodes_list = []
        for t in tokens:
            t = t.upper()
            t = re.sub("[^A-Z0-9]+", "", t) # 영문/숫자만 남김    
            if len(t) == 0: continue
            s = tcode_pattern.search(t)
            if s != None:
                tcode = t[s.start():s.end()]
                if tcode not in tcodes_list:
                    tcodes_list.append(tcode)      
            
            # 정식 네이밍룰을 벗어나는 경우도 있으므로, Z로 시작하고 숫자로 끝나면 우선 TCODE로 가정한다.(최종 판단은 프로그램 마스터 기준으로 RFC에서 할 것임)
            elif t[0] == 'Z' and len(t) >= 5 and len(t) <= 30:
                if t[len(t)-1] in num_list or t[len(t)-2] in num_list: # Main. View처럼 M으로 끝나는 경우도 잇어서 끝에서 1/2자리를 비교함
                    tcode = t
                    if tcode not in tcodes_list:
                        tcodes_list.append(tcode)
        
        # / 로 합쳐서 문자열로 반환
        if len(tcodes_list) > 0:            
            return'/'.join(tcodes_list)
        else:
            return ''
