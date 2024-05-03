import json
# from paddlenlp import Taskflow
from tqdm import tqdm
import random
from os.path import join

def clean_data(origin_data_path, target_data_path):
    with open(origin_data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    new_data = []
    for line in data:
        if line == '\n':
            continue
        new_data.append(line)
    # 取出偶数行
    new_data = new_data[1::2]
    # 去除&nbsp;
    new_data = [line.replace('&nbsp;', '') for line in new_data]
    # 去除\n
    new_data = [line.replace('\n', '') for line in new_data]
    # 保存为txt文件，每行为一条数据
    with open(target_data_path, mode='w+', encoding='utf-8') as f:
        for line in new_data:
            f.write(line + '\n')

def convert(result):
    result = result[0]
    formatted_result = []
    for label, ents in result.items():
        for ent in ents:
            formatted_result.append(
                {
                    "label": label,
                    "start_offset": ent['start'],
                    "end_offset": ent['end']
                })
    return formatted_result

# 自动识别中文命名实体
# def auto_recognize_entity(txt_data_path, json_temp_data_path, schema=['时间', '地点', '人名']):
#     ie = Taskflow('information_extraction', schema=schema)
#     data_dict = []
#     with open(txt_data_path, 'r', encoding='utf-8') as f:
#         data = f.readlines()
#     # 读取每条数据，识别实体
#     for line in tqdm(data):
#         result = ie(line)
#         formatted_result = convert(result)
#         formatted_result.sort(key=lambda x: x['start_offset'])
#         data_dict.append({
#             'text': line,
#             'entities': formatted_result
#         })
#     with open(json_temp_data_path, mode='w+', encoding='utf-8') as f:
#         json.dump(data_dict, f, ensure_ascii=False, indent=2)

# 将自动识别的实体转化为规定的json格式
def convert_format(json_temp_data_path, json_data_path):
    with open(json_temp_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    for data_item in data:
        content = data_item['text']
        entities = data_item['entities']
        records = []
        for entity in entities:
            offset = [entity['start_offset'], entity['end_offset'] - 1] # end_offset - 1，变为左闭右闭
            tag = entity['label']
            records.append({
                'offset': offset,
                'tag': tag
            })
        new_data.append({
            'content': content,
            'records': records
        })
    with open(json_data_path, mode='w+', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

# 将自动识别的实体转化为规定的json格式
def convert_format2(json_temp_data_path, json_data_path, schema_kv):
    with open(json_temp_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    for data_item in data:
        content = list(data_item['content'])
        records = data_item['records']
        entities = []
        for record in records:
            start_offset = record['offset'][0]
            end_offset = record['offset'][1]
            tag = record['tag']
            span = record["span"]
            entities.append({
                'start_offset': start_offset,
                'end_offset': end_offset,
                'label': schema_kv[tag],
                'span': span
            })
        new_data.append({
            'text': content,
            'entities': entities
        })
    with open(json_data_path, mode='w+', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


# 读取target_result2.txt文件，对每条数据里的entity里的start_offset和end_offset，增加一个content字段
def add_content(target_data_path, target2_data_path):
    with open(target_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for data_item in data:
        text = data_item['text']
        entities = data_item['entities']
        for entity in entities:
            entity['content'] = text[entity['start_offset']:entity['end_offset']]
    with open(target2_data_path, mode='w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 读取json文件, 将数据随机分为训练集、验证集、测试集，比例为14:3:3
def splitData(origin_data, span_data_path):
    with open(origin_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 打乱数据，设置随机种子
    random.seed(42)
    random.shuffle(data)
    # 分割数据
    train_data = data[:int(len(data) * 0.7)]
    dev_data = data[int(len(data) * 0.7):int(len(data) * 0.85)]
    test_data = data[int(len(data) * 0.85):]
    # 保存数据
    with open(span_data_path + 'train.json', mode='w+', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(span_data_path + 'dev.json', mode='w+', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(span_data_path + 'test.json', mode='w+', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 读取span数据，将数据转为序列标注格式
def convert2sequence(span_data_path, sequence_data_path, markup='bio'):
    sequence_data_path = join(sequence_data_path, markup)
    if markup == 'bio':
        # 读取数据, 将其转化为bio标注格式
        for data_type in ['train.json', 'dev.json', 'test.json']:
            data_path = join(span_data_path, data_type)
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            new_data = []
            for data_item in data:
                text = list(data_item['text'])
                entities = data_item['entities']
                sequence = ['O'] * len(text)
                for entity in entities:
                    start_offset = entity['start_offset']
                    end_offset = entity['end_offset']
                    label = entity['label']
                    sequence[start_offset] = 'B-' + label
                    for i in range(start_offset + 1, end_offset + 1):
                        sequence[i] = 'I-' + label
                new_data.append({
                    'text': text,
                    'label': sequence
                })

            save_data_path = join(sequence_data_path, data_type)
            with open(save_data_path, mode='w+', encoding='utf-8') as f:
                for item in new_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
    elif markup == 'bmeso':
        # 读取数据, 将其转化为bmeso标注格式
        for data_type in ['train.json', 'dev.json', 'test.json']:
            data_path = join(span_data_path, data_type)
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            new_data = []
            for data_item in data:
                text = list(data_item['text'])
                entities = data_item['entities']
                sequence = ['O'] * len(text)
                for entity in entities:
                    start_offset = entity['start_offset']
                    end_offset = entity['end_offset']
                    label = entity['label']
                    if start_offset == end_offset:
                        sequence[start_offset] = 'S-' + label
                    else:
                        sequence[start_offset] = 'B-' + label
                        sequence[end_offset] = 'E-' + label
                        for i in range(start_offset + 1, end_offset):
                            sequence[i] = 'M-' + label
                new_data.append({
                    'text': text,
                    'label': sequence
                })

            save_data_path = join(sequence_data_path, data_type)
            with open(save_data_path, mode='w+', encoding='utf-8') as f:
                for item in new_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

if __name__ == '__main__':
    origin_data_path = "../datasets/origin_data/origin_result.txt"
    txt_data_path = "../datasets/origin_data/txt_result.txt"
    json_temp_data_path = "../datasets/origin_data/json_temp_result.json"
    json_data_path = "../datasets/origin_data/json_result.json"
    # span_data(origin_data_path, txt_data_path)

    schema = ['时间', '地点', '人名']
    schema_kv = {'时间': 'time', '地点': 'loc', '人名': 'person'}

    # 自动识别实体
    # auto_recognize_entity(txt_data_path, json_temp_data_path, schema)
    # 转化为规定的json格式
    # convert_format(json_temp_data_path, json_data_path)

    # add_content(target_data_path, target2_data_path)

    origin_data = "../datasets/origin_data/text_entity_extraction.json"
    origin_tmp_data = "../datasets/origin_data/tmp_entity_extraction.json"
    span_data_path = "../datasets/span_data/"
    seq_data_path = "../datasets/seq_data/"
    markup = "bmeso"

    convert_format2(origin_data, origin_tmp_data, schema_kv)
    splitData(origin_tmp_data, span_data_path)

    # convert2sequence(span_data_path, seq_data_path, markup)
