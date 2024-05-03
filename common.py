#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/25 16:26 
# ide： PyCharm
import pandas as pd
import erniebot
from tqdm import tqdm

from db import milvusClient
import time
import datetime
import logging

logging.basicConfig(level=logging.INFO)
mc = milvusClient()
mc.createCollection()
mc.load()

erniebot.api_type = "aistudio"
erniebot.access_token = "fae6dc75253e70e6826642608deb3abcc68ae5d3"


def embedding(token, text):
    erniebot.access_token = token
    response = erniebot.Embedding.create(
        model="ernie-text-embedding",
        input=[text])

    return response.get_result()

def chat(token, model, text, temperature, top_p):
    erniebot.access_token = token
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        temperature=temperature,
        top_p=top_p)
    return response.get_result()

def upload_and_create_vector_store(file, token):
    filepath = file.name
    if filepath.endswith(".csv"):
        try:
            data_df = pd.read_csv(filepath,encoding='gbk')
        except:
            data_df = pd.read_csv(filepath, encoding="utf-8")
    elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        data_df = pd.read_excel(filepath)
    elif filepath.endswith(".txt"):
        data_df = pd.read_table(filepath, header=None)

    reply = data_df.iloc[:, 0]
    for rly in tqdm(reply):
        try:
            rlyEmbedding = embedding(token, rly)
            mc.insert(rly, rlyEmbedding)
            time.sleep(0.5)
        except:
            pass
    logging.info("【数据导入】成功导入文本向量")
    return "成功导入文本向量"

def add_text(history, text):
    history = history + [[text, None]]
    return history, ""

def bot(history,
        llm,
        token,
        instruction,
        temperature,
        top_p,
        ):

    text = history[-1][0]
    logging.info("【用户输入】 {}".format(text))
    text_embedding = embedding(token, text)
    ragResult = mc.search(text_embedding)
    erniebotInput = instruction + "用户最后的问题是：" + text + "。给定的文段是：" + ragResult
    logging.info("【文心Prompt】 {}".format(erniebotInput))
    chatResult = chat(token, llm, erniebotInput, temperature, top_p)
    logging.info("【文心Answer】 {}".format(chatResult))
    history[-1][1] = chatResult
    return history


