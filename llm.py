import requests
import json

class llm:
    
    def prompt(self, urlbase, model, prompt, temperature):
        try: 
            url = urlbase + "/generate"
            params = {"model": model,
                    "prompt": prompt, 
                    "stream": False,
                    "temperature": float(temperature.replace(",", "."))}
            response = requests.post(url, json=params)
            if (response.status_code == 200):
                response_text = response.text
                data = json.loads(response_text)
                return data["response"]
            else:
                raise Exception("Error while reaching out the Web Service: {}", str(response.status_code, response.text))
        except Exception as e:
            print(e)
            return str(e)