import argparse
import time
from datetime import timedelta
from document import document
from similaritySearch import similaritySearch
from embeddingsFactory import embeddingsFactory
from llm import llm

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
        print("Text length : {}".format(len(pdf.content)))
        print("PDF to TEXT conversion -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 2 - Chunk document
        if (len(pdf.content) > 0):
            nb, chunks = pdf.chunk(args["separator"], args["chunk_size"], args["chunk_overlap"])
        print("Number of chunks : {}".format(nb))
        print("Chunking -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 3 - Text embeddings
        vPrompt = embeddingsFactory.createEmbeddingsFromTXT(args["prompt"])
        print("Create embeddings from prompt -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 4 - Chunks embeddings
        vChunks = embeddingsFactory.createEmbeddingsFromJSON(chunks)
        print("Create embeddings from chunks -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 5 - Similarity Search
        myfaiss = similaritySearch()
        myfaiss.loadChunks(vChunks)
        myfaiss.buildIndexFlatL2()
        vtText = myfaiss.loadText(vPrompt)
        similars = myfaiss.getNearest(vtText, 3)
        print("Similarity Search -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 6 - Build prompt
        promptTemplate = "Question: {prompt}\n please answer the question based on the informations listed below: info0: {info0}\ninfo1: {info1}\ninfo2: {info2}"
        prompt = promptTemplate.format(prompt=args["prompt"],
                                       info0=similars["chunk"][0],
                                       info1=similars["chunk"][1],
                                       info2=similars["chunk"][2])
        print("Prompt build -> {}".format(str(timedelta(seconds=time.perf_counter() - start))))
        
        # 7 - Ask to the LLM ...
        myllm = llm()
        resp = myllm.prompt(args["urlbase"], args["model"], prompt, args["temperature"])
        print("<RESPONSE>\n {}\n<END OF RESPONSE>".format(resp))
        print("LLM Q&A : {}".format(str(timedelta(seconds=time.perf_counter() - start))))
    
    except Exception as e:
        print(e)