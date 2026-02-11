import requests

url="https://dblp.org/search/publ/api?q=graph+neural+network&format=json&h=10&f=0"
url='https://api.springernature.com/meta/v2/pam?api_key=832dbae15193797443b7e0653891c37b&callback=&s=1&p=10&q=(title:"steganography")'
response = requests.request("GET", url)

print(response.text)