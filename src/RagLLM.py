import argparse
from utils.traceOut import traceOut
import utils.functions as F
import utils.CONST as C

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-" + C.ARG_PROMPT[0], help=C.ARG_PROMPT[1], required=True)
        parser.add_argument("-" + C.ARG_MODEL[0], help=C.ARG_MODEL[1], required=False, default="tinydolphin")
        parser.add_argument("-" + C.ARG_URL[0], help=C.ARG_URL[1], required=False, default="http://localhost:11434/api")
        parser.add_argument("-" + C.ARG_TEMP[0], help=C.ARG_TEMP[1], required=False, type=float, default=0.9)
        args = vars(parser.parse_args())
        myTrace = traceOut(args)
        myTrace.start()

        resp = F.promptLLM(myTrace, args[C.ARG_PROMPT[0]], args[C.ARG_URL[0]], args[C.ARG_MODEL[0]], args[C.ARG_TEMP[0]])

        F.wrapTrace(resp)
        F.wrapResponse(C.OUT_SUCCESS)
        
    except Exception as e:
        F.wrapResponse(C.OUT_ERROR)
        F.wrapTrace(str(e))

if __name__ == "__main__":
    main()