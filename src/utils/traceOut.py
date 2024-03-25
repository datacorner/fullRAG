import time
from datetime import timedelta, datetime
import json

class traceOut:
    def __init__(self, args):
        self.perfCounter = None
        self.startTime = None
        self.stopTime = None
        self.traceSteps = []
        self.traceHeader = {}
        self.stepIdx = 1
        self.traceHeader = {}
        for arg in args.keys():
            self.traceHeader[arg] = args[arg]

    def start(self):
        if (self.perfCounter == None):
            self.perfCounter = time.perf_counter()
            self.startTime = datetime.now()

    def add(self, name, description, *others) -> bool:
        try:
            if (self.perfCounter == None):
                self.start()
            curTrace = {}
            curTrace["step"] = self.stepIdx
            curTrace["name"] = name
            curTrace["description"] = description
            curTrace["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            curTrace["stepduration"] = str(timedelta(seconds=time.perf_counter() - self.perfCounter))
            if (len(others) > 0):
                curTrace["details"] = others
            self.traceSteps.append(curTrace)
            self.stepIdx = self.stepIdx + 1
            return True
        except Exception as e:
            return False
        
    def stop(self):
        self.stopTime = datetime.now()

    def getFullJSON(self):
        fullJson = {}
        fullJson["parameters"] = self.traceHeader
        fullJson["steps"] = self.traceSteps
        fullJson["start"] = str(self.startTime)
        fullJson["stop"] = str(self.stopTime)
        fullJson["duration"] = str(self.stopTime - self.startTime)
        return json.dumps(fullJson)