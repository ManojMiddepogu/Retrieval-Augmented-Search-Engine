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
"GOOGLE_CSE_ID" = <> <br>
"GOOGLE_API_KEY" = <> <br>
Please refer to this [link](https://python.langchain.com/docs/integrations/tools/google_search) on how to get the above keys.<br>
Update in **search_zero_shot.py** and **search_fine_tuned.py** 


## Method1 (Retrieval Augmented zero-shot Generation)
```bash
python search_zero_shot.py "your query goes here"
```

## Method2 (Retrieval Augmented fine-tuned Generation)
Download all files in [drive](https://drive.google.com/drive/folders/1GYJSihfUMR01TZBdAKbY8kkkw4KIZLtx) and place in mistral-we-com3 folder itself.
```bash
python search_fine_tuned.py "your query goes here"
```

## Design
### Method1 (Retrieval Augmented zero-shot Generation)
In short, this method retrieves the data relevant to the query from top 10 web pages given by Google search API. Then the text is made into chunks and the embeddings are computed using embedding transformers and stored in a database. FAISS or ChromaDB both can be used to store and retrieve, i went ahead with the FAISS for the better score measure. We can create a db over chunks of WiKi and retrieve content from Wiki like this and use a LLM model to ask questions if we want to only interact with Wiki, i went a step further and tried to build a generic search tool prototype.

Once the text is retrieved, it is feeded into the prompt of a large language model. Here I've used mistralai/Mistral-7B-Instruct-v0.2, this is atleast as better as Llama2-7b both chat and Instruct models and closer to performance of Llama2 13B model on open benchmarks, so went ahead with that. This models P(y | X, context)

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

The context above is restricted to 4000 absolute length because of OOM issues and other model issues. In actual if large context has to be passed, we can do sort of map-reduce way of feeding context into the LLM and get a final answer. This framework has been already developed by Langchain. Since this is a prototype, i am stating it here but it is easy to plug-in at the expense of more latency. 

### Method2 (Retrieval Augmented fine-tuned Generation)
In short, this method's retrieval part is same as the above.

To improve model's ability to attend to both context and its knowledge for this representation, i tried to fine-tune the mistralai/Mistral-7B-Instruct-v0.2 model. I used the dolly-15k dataset where context, question and response is provided. I did a PEFT finetuning with 4-bit loading. More low level details are in [add link]

The same above prompt is used here.

## Improvements
* One straight forward improvement is to collect more data <context, question, answer> triplets and do more fine-tuning effectively.
* Alignment - preference/non-preference. Datasets like WebGPT and many datasets [add link] here have <context, question, answer1, answer2> collections, where we can do PPO/DPO and align to already rated/GPT4 responses.
* Next is differentiable search index, which can bridge gap between retrieval and generation.
* Also improving factuality should be important. On some queries I tested the model is mixing up the facts. So staying grounded to context is the least ability the model should have. There has have been research in this like contexual decoding[add link] which can improve factuality.  

## Evaluation
Regarding evaluation, basic evals can be done on HotpotQA, BeerQA (more general) and StrategyQA. Is computationally expensive (so avoided for this prototype) but easy to implement and verify.




