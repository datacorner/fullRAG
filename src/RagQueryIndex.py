import argparse
from elements.embeddingsFactory import embeddingsFactory
from utils.traceOut import traceOut
import utils.functions as F
from elements.FAISSWrapper import FAISSWrapper
import utils.CONST as C

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PROMPT[0], help=C.ARG_PROMPT[1], required=True)
        parser.add_argument("-" + C.ARG_TEMP[0], help=C.ARG_TEMP[1], required=False, type=float, default=0.9)
        parser.add_argument("-" + C.ARG_NEAREST[0], help=C.ARG_NEAREST[1], required=False, type=int, default=3)
        parser.add_argument("-" + C.ARG_MODEL[0], help=C.ARG_MODEL[1], required=False, default="tinydolphin")
        parser.add_argument("-" + C.ARG_URL[0], help=C.ARG_URL[1], required=False, default="http://localhost:11434/api")
        parser.add_argument("-" + C.ARG_FAISSNAME[0], help=C.ARG_FAISSNAME[1], required=True)
        parser.add_argument("-" + C.ARG_FAISSPATH[0], help=C.ARG_FAISSPATH[1], required=True)
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        # 1 - Text embeddings
        embFactory = embeddingsFactory()
        vPrompt = F.textEmbeddings(myTrace, embFactory, args[C.ARG_PROMPT[0]])
        # 2 - Load the existing index
        myfaiss = FAISSWrapper()
        F.FAISSLoad(myTrace, myfaiss, args[C.ARG_FAISSPATH[0]], args[C.ARG_FAISSNAME[0]])
        # 3 - Similarity Search
        similars = F.FAISSSearch(myTrace, myfaiss, args[C.ARG_NEAREST[0]], vPrompt)
        # 4 - Build prompt
        customPrompt = F.buildPrompt(myTrace, args[C.ARG_PROMPT[0]], similars["text"])
        # 5 - Ask to the LLM ...
        resp = F.promptLLM(myTrace, customPrompt, args[C.ARG_URL[0]], args[C.ARG_MODEL[0]], args[C.ARG_TEMP[0]])

        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(resp)
        F.wrapStatusOK()

    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapError(str(e))
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapStatusERROR()
        
if __name__ == "__main__":
    main()