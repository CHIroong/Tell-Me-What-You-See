import json

from collections import defaultdict

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

class DOMFeatureExtracter:
    def __init__(self, filename, easylist=None, tagger=None, lexrank=None):
        self.soup = BeautifulSoup(open(filename, "r", encoding='UTF-8').read(), 'html.parser')
        self.filename = filename
        self.point_data = json.loads(self.soup.body["sg:point-data"])
        self.easylist = easylist
        self.okt = tagger
        self.lexrank = lexrank

    def size(self):
        box = json.loads(self.soup.body["sg:rect"])
        return len(self.point_data[0]) * 16, len(self.point_data) * 16
        #return box["width"], box["height"]

    def get_point_data(self, x, y):
        x -= x % 16
        y -= y % 16
        x_max = len(self.point_data[0]) - 1
        y_max = len(self.point_data) - 1
        return self.point_data[min(y//16, y_max)][min(x//16, x_max)]
    
    def get_element_by_id(self, sg_id):
        elem = self.soup.find(attrs={"sg:id": sg_id})
        while elem is not None and ("sg:style" not in elem or "sg:rect" not in elem):
            elem = elem.parent
        return elem if elem is not None else self.soup.body
    
    def get_element_by_point(self, x, y):
        return self.get_element_by_id(self.get_point_data(x, y))
    
    def cursor_style(self, x, y):
        elem = self.get_element_by_point(x, y)
        return json.loads(elem["sg:style"])["cursor"]

    def get_bounding_box(self, elem):
        return json.loads(elem["sg:rect"])
    
    def element_aspect_ratio(self, x, y):
        elem = self.get_element_by_point(x, y)
        box = self.get_bounding_box(elem)
        while box["width"] < 15 or box["height"] < 15:
            elem = elem.parent
            box = self.get_bounding_box(elem)
        return box["width"] / box["height"]
    
    def is_img(self, x, y):
        elem = self.get_element_by_point(x, y)
        return elem.name == 'img'
    
    def is_iframe(self, x, y):
        elem = self.get_element_by_point(x, y)
        return elem.name == 'iframe'
    
    def num_nested_a_tags(self, x, y):
        elem = self.get_element_by_point(x, y)
        count = 0
        while elem.parent is not None:
            if elem.name == 'a':
                count += 1
            elem = elem.parent
        return count
    
    def has_harmful_url_segment(self, x, y):
        elem = self.get_element_by_point(x, y)
        while elem.parent is not None:
            if elem.name == 'a' and "href" in elem:
                if self.easylist.is_harmful_url(elem["href"]):
                    return True
            elif elem.name in ['img', 'iframe'] and "src" in elem:
                if self.easylist.is_harmful_url(elem["src"]):
                    return True
            elem = elem.parent
        return False
    
    def text_with_styles(self, elem):
        if type(elem) == NavigableString and len(str(elem).strip()) > 0 and \
           elem.parent.name in "h1 h2 h3 h4 h5 h6 p span div":
            return [(json.loads(elem.parent["sg:style"]), str(elem))]
        ret = []
        for child in elem.children:
            try:
                ret += self.text_with_styles(child)
            except Exception:
                continue
        return ret

    @staticmethod
    def z_score_of(value_and_amount):
        total = sum(amount for value, amount in value_and_amount)
        mean = sum(value * amount for value, amount in value_and_amount) / total
        var = sum((value - mean)**2 * amount for value, amount in value_and_amount) / total
        def z(x):
            return (x - mean) / var ** 0.5
        return z

    def salient_keywords(self):
        text_and_styles = self.text_with_styles(self.soup.body)
        word_to_weight = defaultdict(lambda:{"size":0, "weight": 0})
        font_sizes = defaultdict(lambda:0)
        font_weights = defaultdict(lambda:0)
        for style, text in text_and_styles:
            for word in list(map(lambda x:x[0], filter(lambda i:i[1] in ('Noun', 'Verb', 'Adjective'), self.okt.pos(text)))):
                font_sizes[float(style["font-size"].replace("px", ""))] += len(word)
                font_weights[float(style["font-weight"])] += len(word)
                word_to_weight[word]["size"] = float(style["font-size"].replace("px",""))
                word_to_weight[word]["weight"] = float(style["font-weight"])
        

        word_to_score = defaultdict(lambda:0)
        for word, weights in word_to_weight.items():
            word_to_score[word] += weights["size"] #* (2 if weights["weight"] >= 600 else 1)
        
        return [word for word, score in sorted(word_to_score.items(), key=lambda x:-x[1])[:4]]
    
    def salient_sentences(self):
        text_and_styles = self.text_with_styles(self.soup.body)
        text_and_styles.sort(key=lambda x:-float(x[0]["font-size"].replace("px","")))
        return [text for style, text in text_and_styles[:5]]
    
    def summary(self):
        try:
            texts = []
            for style, text in self.text_with_styles(self.soup.body):
                for _text in text.split('.'):
                    texts.append(_text.strip().strip("."))
            text = ". ".join(texts)
            self.lexrank.summarize(text)
            return self.lexrank.probe(5)
        except Exception as e:
            print(e)
            return []
    
    def num_tags(self, tagnames):
        ret = 0
        for tagname in tagnames.split(","):
            ret +=  len(self.soup.find_all(tagname))
        return ret

    def title(self):
        if self.soup.find("title") is not None:
            return self.soup.find("title").text
        return ""

    def navs(self):
        if self.soup.find('nav') is not None:
            return [li.text for li in self.soup.find('nav').find_all('li')[:5]]
        return []
    
    def headers(self):
        ret = []
        ret += [i.text for i in self.soup.find_all('h1')]
        ret += [i.text for i in self.soup.find_all('h2')]
        ret += [i.text for i in self.soup.find_all('h3')]
        ret += [i.text for i in self.soup.find_all('h4')]
        ret += [i.text for i in self.soup.find_all('h5')]
        ret += [i.text for i in self.soup.find_all('h6')]
        return ret[:10]

    """
    def has_harmful_css_class(self, x, y):
        elem = self.get_element_by_point(x, y)
        while elem.parent is not None:
            if "class" in elem and self.easylist.is_harmful_css_classes(elem["class"]):
                return True
            elem = elem.parent
        return False

    def has_harmful_css_id(self, x, y):
        elem = self.get_element_by_point(x, y)
        while elem.parent is not None:
            if "id" in elem and self.easylist.is_harmful_css_id(elem["id"]):
                return True
            elem = elem.parent
        return False
    """

if __name__ == "__main__":
    dom = DOMFeatureExtracter("raw_data/2.txt")
    x, y = (873, 2627)
    print(dom.get_element_by_point(x,y))
    print(dom.cursor_style(x, y))
    print(dom.element_aspect_ratio(x, y))
    print(dom.is_img(x, y))
    print(dom.is_iframe(x, y))
    print(dom.num_nested_a_tags(x, y))
    print(dom.has_harmful_url_segment(x, y))