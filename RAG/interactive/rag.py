import pandas as pd
import numpy as np
import os
import json
import obonet
import inflect
import networkx as nx
import matplotlib.pyplot as plt
import requests
import re
from openai import OpenAI
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import ast

api_key =
# import os
# os.environ["OPENAI_API_KEY"] = 
client = OpenAI(api_key = api_key)

# To replace the legacy openai.embeddings_utils.get_embedding function
def get_embedding(text, model="text-embedding-3-small"):
   text = str(text).replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
# 加載 RAG 資料庫
cell_type_db = pd.read_csv("lung_canonical_cell_type_markers_from_literature_v2.csv")
marker_importance_db = pd.read_csv("lung_cluster_cell_type_marker_importance_v2.csv")
# 沒做上面的話，直接跑這個
cell_type_db["embedding"] = pd.read_csv("cell_type_embedding_v2.csv")["embedding"]
cell_type_db["embedding"] = cell_type_db["embedding"].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))

# Alternate the lagecy openai.embeddings_utils.search_vectors function
def cosine_similarity(a, b):
    # print(a, b, type(a), type(b), type(a[0]), type(b[0]))
    # print(np.linalg.norm(a), np.linalg.norm(b))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def annotate_GPT_interact (tissue: str, markers: str, n=5) -> str:
    global client
    responses = []


    # 提取輸入的 marker
    # print(markers, type(markers))
    input_markers = markers.split(", ")
    input_embedding = get_embedding(", ".join(input_markers))
    cell_type_db["similarity"] = cell_type_db["embedding"].apply(lambda x: cosine_similarity(x, input_embedding))
    # marker_importance_db["similarity"] = marker_importance_db["embedding"].apply(lambda x: cosine_similarity(x, input_embedding))

    # 進行檢索
    cell_type_results = (
        cell_type_db.sort_values("similarity", ascending=False)
        .head(n)
    )
    # marker_importance_results = (
    #     marker_importance_db.sort_values("similarity", ascending=False)
    #     .head(n)
    # )

    # 組合上下文
    context = "### Cell Type Database Results ###\n"
    for _,  result in cell_type_results.iterrows():
        
        context += f"Cell Type: {result['cell type']}\nMarkers: {result['markers']}\n"
        tempdb = marker_importance_db[marker_importance_db["cell_type"] == result["cell type"]]
        if not tempdb.empty:
            for marker in result['markers'].split(", "):
                imp = tempdb[tempdb["marker"] == marker]
                if not imp.empty:
                    context += f"Importance of {marker}: {imp['importance'].values[0]}\n"
                else:
                    context += f"Importance of {marker}: Not Found\n"
        context += "\n"

    # 提示用戶
    markers = ("### task: Identify cell types ###\n" + markers + '\n'
    + "Answer is not necessary to be in the Database. "
    + "Only provide the cell type name, formatted as: {{cell_type_name}}. "
    # + "Do not show numbers before the name. "
    + "Some can be a mixture of multiple cell types. "
    # + "You cannot answer unknown. Guess if you have to. "
    # + "If you don't know the answer, output{{-1}}"
    )
    # 提供給 GPT 模型
    retry = 0
    while retry < 3:
        completion = client.chat.completions.create(
            model="gpt-4o",
            markerss=[
                {"role": "system", "content": f"Identify cell types of {tissue} tissue using the markers provided. Use the retrieved context for your decision."},
                {"role": "user", "content": f"{context}\n\n{markers}"}
            ],
            # response_format=cellTypeFormat
        )
        # print(completion.choices[0].markers.parsed)
        print(completion.choices[0].markers.content)
        candi = completion.choices[0].markers.content
        candi = re.findall(r"\{\{(.*?)\}\}", candi)
        if (not candi) or "unknown" in candi[0] or "Unknown" in candi[0]:
            retry += 1
        else:
            responses = candi[0]
            break

    # fail to get a valid answer, try without reg
    if retry == 3:
        markers = ", ".join(input_markers)
        completion = client.chat.completions.create(
            model="gpt-4o",
            markerss=[
                {"role": "system", "content": f"Identify cell types of {tissue} tissue using the markers provided. "
                + "Only provide the cell type name, formatted as: {{cell_type_name}}. "
                + "Do not show numbers before the name. "
                + "Some can be a mixture of multiple cell types."
                # + "Don't answer unknown. If must, formatted as: {{-1}}"
                },
                {"role": "user", "content": markers}
            ],
            # response_format=cellTypeFormat
        )
        print(completion.choices[0].markers.content)
        candi = completion.choices[0].markers.content
        candi = re.findall(r"\{\{(.*?)\}\}", candi)
        if candi and candi[0] != "-1":
            responses = candi[0]
        else:
            responses = "Unknown"

    return responses

if __name__ == "__main__":
    tissue = input("Enter tissue name: ")
    markers = input("Enter markers: ")
    print(annotate_GPT_interact(tissue, markers))