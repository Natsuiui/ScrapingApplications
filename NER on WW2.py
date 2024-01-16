from transformers import pipeline
from bs4 import BeautifulSoup
import requests
HEADERS = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \ AppleWebKit/537.36 (KHTML, like Gecko) \ Chrome/90.0.4430.212 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'})

html_raw = requests.get("https://en.wikipedia.org/wiki/World_War_II", headers=HEADERS).text
soup = BeautifulSoup(html_raw, "lxml")
data = []
data_str = ""
for item in soup.find_all("p"):
    data_str = data_str + item.get_text()
    data.append(data_str)
    data_str = ""

text = [i[:len(i)-1] for i in data if i != "\n" and len(i) >= 100]

ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
for i in text:
    pred = ner(i)[0]
    if pred['entity'] == 'I-PER':
        print(f"Name - {pred['word']}")
    elif pred['entity'] == 'I-LOC':
        print(f"Location - {pred['word']}")