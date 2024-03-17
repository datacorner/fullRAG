import argparse
import time
from datetime import timedelta
from document import document
from similaritySearch import similaritySearch
from embeddingsFactory import embeddingsFactory
from llm import llm

def trace(text):
    print("[{}] {}".format(str(timedelta(seconds=time.perf_counter() - start)), text))

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-prompt", help="Prompt to send to LLAMA2", required=True)
        parser.add_argument("-pdf", help="PDF file path", required=True)
        parser.add_argument("-temperature", help="LLM Temperature", required=False, default=0.9)
        parser.add_argument("-chunk_size", help="Chunk Size", required=False, default=500)
        parser.add_argument("-chunk_overlap", help="Chunk Overlap", required=False, default=50)
        parser.add_argument("-separator", help="Separator", required=False, default=".")
        parser.add_argument("-model", help="Ollama Model", required=False, default="tinydolphin")
        parser.add_argument("-urlbase", help="URL for OLLAMA (default localhost)", required=False, default="http://localhost:11434/api")
        args = vars(parser.parse_args())
        
        start = time.perf_counter()
        
        # 1 - Read the pdf content
        pdf = document(args["pdf"])
        pdf.getContentFromPDF()
        trace("Text length : {}".format(len(pdf.content)))
        trace("PDF converted to TEXT successfully")
        
        # 2 - Chunk document
        if (len(pdf.content) > 0):
            nb, chunks = pdf.chunk(args["separator"], args["chunk_size"], args["chunk_overlap"])
        trace("Document chunked successfully, Number of chunks : {}".format(nb))
        
        # 3 - Text embeddings
        vPrompt = embeddingsFactory.createEmbeddingsFromTXT(args["prompt"])
        trace("Embeddings created from prompt successfully")
        
        # 4 - Chunks embeddings
        vChunks = embeddingsFactory.createEmbeddingsFromJSON(chunks)
        trace("Embeddings created from chunks successfully")
        
        # 5 - Similarity Search
        myfaiss = similaritySearch()
        myfaiss.loadChunks(vChunks)
        myfaiss.buildIndexFlatL2()
        vtText = myfaiss.loadText(vPrompt)
        similars = myfaiss.getNearest(vtText, 3)
        trace("Similarity Search executed successfully")
        
        # 6 - Build prompt
        promptTemplate = "Question: {prompt}\n please answer the question based on the informations listed below: info0: {info0}\ninfo1: {info1}\ninfo2: {info2}"
        prompt = promptTemplate.format(prompt=args["prompt"],
                                       info0=similars["chunk"][0],
                                       info1=similars["chunk"][1],
                                       info2=similars["chunk"][2])
        trace("Prompt built successfully")
        
        # 7 - Ask to the LLM ...
        myllm = llm()
        resp = myllm.prompt(args["urlbase"], args["model"], prompt, args["temperature"])
        trace("LLM Reponse\n {}\n".format(resp))
    
    except Exception as e:
        print(e)