from bs4 import BeautifulSoup
import requests

HEADERS = ({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \ AppleWebKit/537.36 (KHTML, like Gecko) \ Chrome/90.0.4430.212 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'})
url = 'https://www.amazon.in/Apple-iPhone-15-Pro-256/dp/B0CHX5J2ND/ref=sr_1_1_sspa?crid=1XHMN7FQTUAYT&dib=eyJ2IjoiMSJ9.iB2vwuex0kDBGVm9gAEJC9tG7i_zmSnC9oU85ZYa98_-DovWgg4x9JRNdCR8IQsqR-WySTV9F2GzL1mZrQW5U9OrvPIjcC3W-6fr2uDvDC61zcE7ZiNF4XTzk371YobdlgUot2XleudEx22Kos6DPShR71nfAUsVaZ75EKlRz0VI3V14qe_BkCrwX40nkoi5zafGEJPDsG6XUok3wp3nDEEpykyPEVmuEFx9eBb3kCQ.GT48QIU3F3vbkUgDF_Bb0IbQAxCU5Mrgpb2DTqVa9pU&dib_tag=se&keywords=iphone%2B15%2Bpro&qid=1705421554&sprefix=iphone%2B15%2Caps%2C203&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'

html_base = requests.get(url, headers=HEADERS)
soup = BeautifulSoup(html_base.text, 'html.parser')

data_str = ""
cus_list = []
for item in soup.find_all("span", class_="a-profile-name"):
    data_str = data_str + item.get_text()
    cus_list.append(data_str)
    data_str = ""

data_str = ""
for item in soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):
    data_str = data_str + item.get_text()
result = data_str.split("\n")
reviews = [i for i in result if i != ""]
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
for rev in reviews:
    print(f'Review {rev[:25]}... is classified as {classifier(rev)[0]["label"]}')