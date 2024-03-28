import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C

def main():
    myTrace = traceOut()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PROMPT[0], help=C.ARG_PROMPT[1], required=True)
        parser.add_argument("-" + C.ARG_MODEL[0], help=C.ARG_MODEL[1], required=False, default="tinydolphin")
        parser.add_argument("-" + C.ARG_URL[0], help=C.ARG_URL[1], required=False, default="http://localhost:11434/api")
        parser.add_argument("-" + C.ARG_TEMP[0], help=C.ARG_TEMP[1], required=False, type=float, default=0.9)
        args = vars(parser.parse_args())
        myTrace.initialize(args)
        myTrace.start()

        resp = F.promptLLM(myTrace, args[C.ARG_PROMPT[0]], args[C.ARG_URL[0]], args[C.ARG_MODEL[0]], args[C.ARG_TEMP[0]])
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