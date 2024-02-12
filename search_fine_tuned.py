import sys
from googleapiclient.discovery import build
from peft import AutoPeftModelForCausalLM
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_google_search_results(search_query):
    
    def google_search(search_term, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

    GOOGLE_API_KEY = ''
    GOOGLE_SE_ID = ''
    results = google_search(search_query, GOOGLE_API_KEY, GOOGLE_SE_ID, num=3)
    return results

def get_html_content(urls):

    def clean_string(input_string):
        import re
        cleaned_string = re.sub(r'\s+', ' ', input_string).strip()
        unicode_pattern = re.compile('[^\x00-\x7F]+')
        cleaned_string = unicode_pattern.sub('', cleaned_string)
        return cleaned_string

    content = {}
    for url in urls:
      response = requests.get(url)
      html_content = response.text
      soup = BeautifulSoup(html_content, 'html.parser')

      for script in soup(["script", "style"]):
          script.extract()

      text = soup.get_text()

      content[url] = clean_string(text)

    return content

def create_faiss_db(html_content):

  with open('tmp_txt_file.txt', 'w') as f:
    f.write(html_content)
  raw_documents = TextLoader('tmp_txt_file.txt').load()
  text_splitter = RecursiveCharacterTextSplitter()
  documents = text_splitter.split_documents(raw_documents)
  db = FAISS.from_documents(documents, HuggingFaceEmbeddings())

  return db

def llm():
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  peft_model_id = "./mistral-we-com3"
  tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
  tokenizer.pad_token = tokenizer.eos_token
  model = AutoPeftModelForCausalLM.from_pretrained(
      peft_model_id,
      quantization_config=bnb_config,
  )
  return model, tokenizer



def llm_output(model, tokenizer, search_query, retrieved_docs):
  PROMPT = """[INST] You are an assistant for search tasks.
Use the following pieces of retrieved context and your knowledge to answer the search query.
Please provide a direct response to the question without additional comments on the context. 
Use 3 sentences maximum and keep the answer concise.
[/INST]
Search Query: {search_query}
Context: {context}
Answer:
"""
  PROMPT = PROMPT.replace('{search_query}', search_query)
  PROMPT = PROMPT.replace('{context}', retrieved_docs)
  encodeds = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
  model_inputs = encodeds.to('cuda')
  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, temperature = 0.1, top_p = 0.95, top_k=40, repetition_penalty=1.2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return(decoded[0])

def search(search_query, llm, tokenizer):
  results = get_google_search_results(search_query)
  links = [result['link'] for result in results]
  html_content = '\n'.join(get_html_content(links).values())
  faiss_db = create_faiss_db(html_content)
  retrieved_docs = "\n".join([doc.page_content for doc in faiss_db.similarity_search(search_query)])[:2000]
  llm_out = llm_output(model, tokenizer, search_query, retrieved_docs)
  print(llm_out[llm_out.find("Answer:")+len("Answer:"):-4].strip())
  return results, links, html_content, faiss_db, retrieved_docs, llm_out

torch.cuda.empty_cache()
model, tokenizer = llm()

search_query = sys.argv[1]
results, links, html_content, faiss_db, retrieved_docs, llm_out = search(search_query, model, tokenizer)
