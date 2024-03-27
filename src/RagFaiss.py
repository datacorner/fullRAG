import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C
from elements.FAISSWrapper import FAISSWrapper

if __name__ == "__main__":
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
        myTrace = traceOut(args)
        myTrace.start()

        if (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALMSEARCH):
            # Memory search / need chunks,
            myfaiss = FAISSWrapper()
            vChunks = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_CK[0]])
            vPrompt = F.readJsonFromFile(args[C.ARG_EMBEDDINGS_PT[0]])
            F.FAISSaddToIndex(myTrace, myfaiss, vChunks)
            similars = F.FAISSSearch(myTrace, myfaiss, args[C.ARG_NEAREST[0]], vPrompt)
            F.writeToFile(args[C.ARG_CHUNKS[0]], similars["text"].to_json())

        elif (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALISEARCH):
            # Index search
            pass
        elif (args[C.ARG_FAISSACTION[0]] == C.ARG_FAISSACTION_VALSTORE):
            # Calculate and Store index
            pass
        else:
            raise Exception("No action selected, please select search or store")

        myTrace.stop()
        F.wrapTrace(myTrace.getFullJSON())
        F.wrapResponse(C.OUT_SUCCESS)
        
    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapTrace(str(e))