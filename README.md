# Retrieval Augmented Search Engine
## Set-up
```bash
conda create -n you_com python=3.10
conda activate you_com
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install trl
pip install peft
pip install bitsandbytes
pip install transformers -U
pip install langchain
pip install faiss-gpu
pip install datasets
pip install beautifulsoup4
pip install google-api-python-client
pip install sentence-transformers
```

## Google search API Keys
GOOGLE_CSE_ID = <> <br>
GOOGLE_API_KEY = <> <br>
Please refer to this [link](https://python.langchain.com/docs/integrations/tools/google_search) on how to get the above keys.<br>
Update in **search_zero_shot.py** and **search_fine_tuned.py** 

## Warning
- Currently the results are not filtered for Hate/Toxic/Unsafe content.
- No checks for factuality is added
- This is a just a prototype of a search CLI, lot of finetuning each step is missing now.  

## Method1 (Retrieval Augmented zero-shot Generation)
```bash
python search_zero_shot.py "your query goes here"
```

## Method2 (Retrieval Augmented fine-tuned Generation)
Download all files in [drive](https://drive.google.com/drive/folders/1GYJSihfUMR01TZBdAKbY8kkkw4KIZLtx) and place in mistral-we-com3 folder itself.
```bash
python search_fine_tuned.py "your query goes here"
```


### Sample Result
![Example Result](/Example_result.png)

## Design
### Method1 (Retrieval Augmented zero-shot Generation)
- This method retrieves the data relevant to the query from top X web pages given by Google search API.
- Then the text is made into chunks  and the embeddings are computed using embedding transformers(any good HF model like BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2 can be chosen) and stored in a vector operations efficient database.
- FAISS or ChromaDB both can be used to store and retrieve, i went ahead with the FAISS for the better score measure.
- We can create a db over chunks of WiKi and retrieve content from Wiki like this and use a LLM to ask questions if we want to only interact with Wiki, i went a step further and tried to build a generic search tool prototype.
- Once the text is retrieved, it is feeded into the prompt of a LLM. Here I've used mistralai/Mistral-7B-Instruct-v0.2, this is atleast as better as Llama2-7b both chat and Instruct models and closer to performance of Llama2 13B model on open benchmarks, so went ahead with that. This essentially models P(Y | query, context)

The prompt used is here:
```bash
""" [INST] You are an assistant for search tasks.
Use the following pieces of retrieved context and your knowledge to answer the search query.
Please provide a direct response to the question without additional comments on the context.
[/INST]
Search Query: {question}
Context: {context}
Answer: {response} </s>
"""
```
{question} : is the search query <br>
{context}  : is the context retrieved above

The context above is restricted to 5000 absolute length because of OOM issues and other model issues. In actual if large context has to be passed, we can do sort of map-reduce way of feeding context into the LLM and get a final answer. This framework has been already developed by Langchain. Since this is a prototype, i am stating it here but it is easy to plug-in at the expense of more latency. 

### Method2 (Retrieval Augmented fine-tuned Generation)
This method's retrieval part is same as the above.

Though Mistral is instruction fine-tuned, To improve model's ability to attend to both context and its knowledge for this data representation, i tried to fine-tune the mistralai/Mistral-7B-Instruct-v0.2 model. I used the dolly-15k dataset where context, question and response is provided. I did a PEFT finetuning with 4-bit loading. I trained on 1 46GB RTX8000 GPU in NYU HPC and it took 4 hrs for one epoch. I fine-tuned LLama2 too but Mistral is way better at following instructions than LLama2. More low level details are in **llm_finetuning.py**.The same above prompt is used here. 

Some tricky problems to train on single type of dataset like dolly is:
- more epochs leads to overfitting and general questions answers are random
- model fits to lengths of the data seen, if more context is given again the model behavior is a bit different. Since I used <=2k context size in training, i had to use 2k context only in inference too. A problem for thought to me!
- All in all, how to train PEFT models while retaining general performance on other tasks is not trivially answered atleast in the study i did.


## Improvements
* One straight forward improvement is to collect more data <context, question, answer> triplets and do fine-tuning effectively. Good Quality and large data, more diverse distributions is needed for better generalization.
* Alignment - preference/non-preference. Datasets like WebGPT and many datasets [here](https://github.com/glgh/awesome-llm-human-preference-datasets?tab=readme-ov-file) have <context, question, answer1, answer2> collections, where we can do PPO/DPO and align to already rated responses.
* Next is differentiable search indexing/retrieval, which can bridge gap between retrieval and generation automatically. There are few papers coming now since RAG is a popular topic.
* Also improving factuality should be important. On some queries I tested the model is mixing up the facts and is wrong. So staying grounded to context is the least ability the model should have. There has have been research in this like [contexual decoding](https://arxiv.org/abs/2305.14739) which can improve factuality.  

## Evaluation
Regarding evaluation, basic evals can be done on HotpotQA, BeerQA (more general) and StrategyQA. Is computationally expensive (so avoided for this prototype) but easy to implement and verify.




