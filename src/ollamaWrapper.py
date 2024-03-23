import requests
import json

class ollamaWrapper:
    def __init__(self, urlbase, model, temperature):
        self.__modelName = model
        self.__urlbase = urlbase
        self.__temperature = temperature
    
    @property
    def model(self):
        return self.__modelName
    @property
    def urlbase(self):
        return self.__urlbase
    @property
    def temperature(self):
        return self.__temperature
    
    def prompt(self, prompt):
        try:
            url = self.urlbase + "/generate"
            params = {"model": self.model,
                      "prompt": prompt, 
                      "stream": False,
                      "temperature": float(self.temperature.replace(",", "."))}
            response = requests.post(url, json=params)
            if (response.status_code == 200):
                response_text = response.text
                data = json.loads(response_text)
                return data["response"]
            else:
                raise Exception("Error while reaching out to the Web Service: {}", str(response.status_code, response.text))
        except Exception as e:
            return str(e)