from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3

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
    return newtensor

conn = sqlite3.connect('list_data.db')
cursor = conn.cursor()

model_name = r"C:\Users\Asus\Documents\github\Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4"

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


# user: llgdfgdfgldjfdjgdfg
# asistan:

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Asus\Desktop\mysql_example\tokens\qwen")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")

question = "Merhaba! Benim adım Halil İbrahim."





messages = [
    {"role": "system", "content": "Karşındaki kişiye arkadaşlık eden bir arkadaşsın."},
    {"role": "system", "content": "Adın Cuma."},
    {"role": "user", "content": question},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(convert(messages[-1]["content"], tokenizer=tokenizer))
# print(convert(question))

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]


print(convert(generated_ids[0]))

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)










