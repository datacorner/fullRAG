import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C
from elements.FAISSWrapper import FAISSWrapper

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_FAISSACTION[0], help=C.ARG_FAISSACTION[1], required=False)
        parser.add_argument("-" + C.ARG_EMBEDDINGS_PT[0], help=C.ARG_EMBEDDINGS_PT[1], required=False)
        parser.add_argument("-" + C.ARG_EMBEDDINGS_CK[0], help=C.ARG_EMBEDDINGS_CK[1], required=False)
        parser.add_argument("-" + C.ARG_NEAREST[0], help=C.ARG_NEAREST[1], required=False, type=int, default=3)
        parser.add_argument("-" + C.ARG_FAISSNAME[0], help=C.ARG_FAISSNAME[1], required=False)
        parser.add_argument("-" + C.ARG_FAISSPATH[0], help=C.ARG_FAISSPATH[1], required=False)
        parser.add_argument("-" + C.ARG_CHUNKS[0], help=C.ARG_CHUNKS[1], required=False)
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        myfaiss = FAISSWrapper()
        if (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALMSEARCH):
            # Memory search / need -> ARG_EMBEDDINGS_CK / ARG_EMBEDDINGS_PT / ARG_CHUNKS
            vChunks = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_CK[0]])
            vPrompt = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_PT[0]])
            F.FAISSaddToIndex(myTrace, myfaiss, vChunks)
            similars = F.FAISSSearch(myTrace, myfaiss, args[C.ARG_NEAREST[0]], vPrompt)
            F.writeToFile(args[C.ARG_CHUNKS[0]], similars["text"].to_json())

        elif (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALISEARCH):
            # Index search / need -> ARG_EMBEDDINGS_PT / ARG_FAISSNAME / ARG_FAISSPATH
            vPrompt = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_PT[0]])
            F.FAISSLoad(myTrace, myfaiss, args[C.ARG_FAISSPATH[0]], args[C.ARG_FAISSNAME[0]])
            similars = F.FAISSSearch(myTrace, myfaiss, args[C.ARG_NEAREST[0]], vPrompt)
            F.writeToFile(args[C.ARG_CHUNKS[0]], similars["text"].to_json())

        elif (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALSTORE):
            # Calculate and Store index / need -> ARG_EMBEDDINGS_CK / ARG_FAISSNAME / ARG_FAISSPATH
            vChunks = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_CK[0]])
            F.FAISSStore(myTrace,  vChunks,  C.ARG_FAISSPATH[0],  args[C.ARG_FAISSNAME[0]])

        else:
            raise Exception("No action selected, please select search or store")

        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(C.OUT_SUCCESS)
        F.wrapStatusOK()

    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapError(str(e))
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapStatusERROR()
        
if __name__ == "__main__":
    main()