from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import numpy as np


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


npage = 1
Dis = []
while True:
    try:
        print(npage)
        if npage == 75:
            break
        page = "https://prensa.presidencia.cl/discursos.aspx?page={}".format(npage)
        f = simple_get(page)
        html = BeautifulSoup(f, "html.parser")
        name_ant = " "
        for link in html.find_all("a"):
            name = link.get("href")
            if name[:16]=="discurso.aspx?id" and name_ant!=name:
                print(name)
                discurso_path = "https://prensa.presidencia.cl/discurso{}".format(name[8:]) #discurso comunicado
                print(discurso_path)
                discurso_page = simple_get(discurso_path)
                html_2 = BeautifulSoup(discurso_page, "html.parser")
                text_html = html_2.select("#main_ltContenido")[0].getText()
                Dis.append(text_html)
                #text_str = text_html.replace("\r", "").replace("\n", "").replace("\xa0", "")
                name_ant = name
        np.save("discursos.npy", Dis)
        npage += 1
    except:
        break
