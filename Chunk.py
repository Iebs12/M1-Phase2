from flask import Flask, render_template
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex #SimpleKeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
#from llama_index.core import PromptTemplate
from llama_index.core.response.pprint_utils import pprint_response
#from llama_index.retrievers import VectorIndexRetriever
#from llama_index.query_engine import RetrieverQueryEngine
#from llama_index.indices.postprocessor import SimilarityPostprocessor
import logging
from dotenv import load_dotenv
import logging
import os

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)  
load_dotenv()
Open_AI_Key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.chunk_size = 256
def chunk_documents_in_directory(directory_path):
    try:
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        #print("This is the Doc:",documents)
        text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=5)
        chunk = {}
        for doc_id, document in enumerate(documents):
            nodes = text_splitter.get_nodes_from_documents([document])
            chunk[doc_id] = nodes
            #print(f"{doc_id}   {chunk}")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        #keyword_index = SimpleKeywordTableIndex(nodes,storage_context=storage_context,show_progress=True,)
        index = VectorStoreIndex(nodes, storage_context=storage_context,
                                    show_progress=True,)
        #retriever = VectorIndexRetriever(index=index)
        #Query Engine 
        query_engine = index.as_query_engine(response_mode="tree_summarize",verbose=True,)
        response = query_engine.query("What are Avacado's Beneficial properties that are utilized in Cosmetics?")
        #print(f"My Response is: {response}")
        pprint_response(response, show_source=True)
        #streaming_response.print_response_stream()
    except Exception as e:
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    directory_path = r'C:/Users/Tanya.gosain/Documents/AI_Analytics_ChatBot/M1-Relevancy/dummy_pdfs'
    response = chunk_documents_in_directory(directory_path)
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)

