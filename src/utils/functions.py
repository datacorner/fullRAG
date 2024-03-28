from elements.document import document
from elements.FAISSWrapper import FAISSWrapper
from elements.ollamaWrapper import ollamaWrapper
from elements.prompt import prompt
import json
import utils.CONST as C
from numpyencoder import NumpyEncoder

# Output printing via XML tags
def wrapResponse(response):
    print(C.TAG_O_RESPONSE + response + C.TAG_C_RESPONSE)
def wrapTrace(response):
    print(C.TAG_O_LOG + response + C.TAG_C_LOG)
def wrapError(response):
    print(C.TAG_O_LOG + response + C.TAG_C_LOG)
def wrapStatusOK():
    print(C.TAG_O_STATUS + C.OUT_SUCCESS + C.TAG_C_STATUS)
def wrapStatusERROR():
    print(C.TAG_O_STATUS + C.OUT_ERROR + C.TAG_C_STATUS)

def readPDF(trace, pdffile):
    # Read the pdf content
    pdf = document(pdffile)
    pdf.getContentFromPDF()
    if (len(pdf.content) <= 0):
        raise Exception("Error while converting the PDF document to text")
    trace.add("PDF2TXT", "PDF converted to TEXT successfully. Text length : {}".format(len(pdf.content)))
    return pdf

def chunkContent(trace, doc, separator, size, overlap):
    # Chunk document
    nb, chunks = doc.chunk(separator, size, overlap)
    if (nb<=0):
        raise Exception("Error while chunking the document")
    trace.add("CHUNKING","Document chunked successfully, Number of chunks : {}".format(nb), nb)
    return nb, chunks

def textEmbeddings(trace, embFactory, prompt):
    vPrompt = embFactory.createEmbeddingsFromTXT(prompt)
    if (vPrompt == {}):
        raise Exception("Error while creating the prompt embeddings")
    trace.add("PTEMBEDDGS", "Embeddings created from prompt successfully")
    return vPrompt

def chunkEmbeddings(trace, embFactory, chunks):
    vChunks = embFactory.createEmbeddingsFromList(chunks)
    if (vChunks == {}):
        raise Exception("Error while creating the chunks embeddings")
    trace.add("DOCEMBEDDGS", "Embeddings created from chunks successfully")
    return vChunks

def FAISSaddToIndex(trace, myfaiss, vChunks):
    myfaiss.addToIndex(vChunks)
    trace.add("ADDTOINDEX", "Add chunks to the FAISS Index")

def FAISSSearch(trace, myfaiss, k, vPrompt):
    similars = myfaiss.getNearest(vPrompt, k)
    trace.add("SIMILARSEARCH", "Similarity Search executed successfully")
    return similars

def FAISSStore(trace, vChunks, path, name):
    myfaiss = FAISSWrapper()
    myfaiss.addToIndex(vChunks)
    trace.add("FAISSSTORE", "Chunks embeddings indexed and stored successfully")
    myfaiss.save(path, name)
    return myfaiss

def FAISSLoad(trace, myfaiss, path, name):
    trace.add("FAISSSTORE", "Chunks embeddings indexed and stored successfully")
    myfaiss.load(path, name)

def buildPrompt(trace, question, similarText):
    myPrompt = prompt(question, similarText)
    customPrompt = myPrompt.build()
    if (len(customPrompt) == 0):
        raise Exception("Error while creating the prompt")
    trace.add("PROMPT", "Prompt built successfully", customPrompt)
    return customPrompt

def promptLLM(trace, question, urlOllama, model, temperature):
    myllm = ollamaWrapper(urlOllama, model, temperature)
    resp = myllm.prompt(question)
    trace.add("LLMPT", "LLM Reponse\n {}\n".format(resp))
    return resp

def writeToFile(filename, content):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        return False

def writeJsonToFile(filename, jsonContent):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(jsonContent, cls=NumpyEncoder))
        return True
    except Exception as e:
        return False
    
def readJsonFromFile(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        return {}