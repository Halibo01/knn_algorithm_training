from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3
import os


def convert(liste:torch.Tensor|str, tokenizer = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(liste) == str:
        if tokenizer == None:
            raise "herhangi bir tokenizer verilmedi"
        tensor_tokens = torch.tensor(tokenizer.encode(liste), dtype=torch.float64).to(device)
    else:
        tensor_tokens = liste.clone().detach().to(device=device, dtype=torch.float64)
    mean = torch.mean(tensor_tokens)
    std_dev = torch.std(tensor_tokens)
    z_score = (tensor_tokens - mean) / std_dev
    z_min = z_score.min()
    z_mean = z_score.mean()
    z_max = z_score.max()
    newtensor = torch.tensor([z_min, z_mean, z_max], dtype=torch.float64).to(device)
    return newtensor.tolist()

def save_to_db(liste:list, dbname:str = "list_data.db", table_name:str = "qa_table"):
    with sqlite3.connect(dbname) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_z_min FLOAT NOT NULL,
            question_z_mean FLOAT NOT NULL,
            question_z_max FLOAT NOT NULL,
            answer_z_min FLOAT NOT NULL,
            answer_z_mean FLOAT NOT NULL,
            answer_z_max FLOAT NOT NULL
        );
        """)

        cursor.execute("""
        INSERT INTO qa_table (question, answer, question_z_min, question_z_mean, question_z_max, answer_z_min, answer_z_mean, answer_z_max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, liste
        )


def main():

    model_name = r"tokens\qwen"
    model_name = os.path.abspath(model_name)

    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")

    question = "Merhaba! Benim adım Halil İbrahim."





    # messages = [{"role": "system", "content": "Karşındaki kişiye arkadaşlık eden bir arkadaşsın."},{"role": "system", "content": "Adın Cuma."},{"role": "user", "content": question}]

    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # model_inputs = tokenizer([text], return_tensors="pt")

    print(convert(question, tokenizer=tokenizer))
    # print(convert(question))

    # generated_ids = model.generate(**model_inputs, max_new_tokens=512)

    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)] # çıtkı tensor


    # print(convert(generated_ids[0]))

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]










