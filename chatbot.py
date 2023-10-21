import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import (AutoModelForCasualLm
                          ,AutoTokenizer
                          ,AutoConfig
                          ,pipeline
                          ,BitsAndBytesConfig)

class ChatBot:
    def __init__(self):
        self.llm = self.initialize_pipeline()

    def initialize_pipeline(self):
        model_id = 'Trelis/Llama-2-7b-chat-hf-sharded-bf16-5GB' 
        quantization_conf = BitsAndBytesConfig(load_in_4bit=True
                                            ,bnb_4bit_quant_type='fp4'
                                            ,bnb_4bit_compute_dtype=torch.bfloat16
                                            ,bnb_4bit_use_double_quant=True)
        config = AutoConfig.from_pretrained(model_id)
        config.init_device = 'cuda:0'
        tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id,config=config,quantization_config=quantization_conf)
        pipe = pipeline(model=model,tokenizer=tokenizer,task='text-generation',max_new_tokens=512,temperature=0.1,device_map='auto',
                    return_full_text=True,
                    repetition_penalty=1.1)
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def create_vectorstore(self,pdf_text):
        splitter = CharacterTextSplitter(separator='\n',chunk_size=1000,chunk_overlap=200)
        split_text = splitter.split_text(pdf_text)
        embedding =  HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',model_kwargs={"device": "cuda"})
        vectorstore = FAISS.from_splits(split_text,embedding)
        return vectorstore

    def create_chain(self,llm,vectorstore):
        self.mem = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=self.mem)
        return chain

    def update(self,pdfText):
        vectorstore = self.create_vectorstore(pdfText)
        self.chain = self.create_chain(self.llm,vectorstore)
    
    def ask(self,question):
        result = self.chain({'question':question,'chat_history':self.mem})
        return result['answer'].rstrip('\n')
        