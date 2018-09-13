from adblockparser import AdblockRules

class EasyListHandler:
    with open('patch_manager/easylist.txt', 'r', encoding='UTF-8') as f:
        rules = AdblockRules(list(f))
    #def __init__(self):
        #with open('easylist.txt', 'r', encoding='UTF-8') as f:
            #self.rules = AdblockRules(list(f))
    def is_harmful_url(self, url):
        return EasyListHandler.rules.should_block(input())

class EasyListCustomHandler:
    def __init__(self):
        self.harmful_css_ids = set() # 8179
        self.harmful_css_classes = set() # 10790
        self.harmful_url_segments = set() # 8368
        with open('easylist.txt', 'r', encoding='UTF-8') as f:
            for line in f:
                if line.startswith("###"):
                    self.harmful_css_ids.add(line[3:])
                if line.startswith("##."):
                    self.harmful_css_classes.add(line[3:])
                for prefix in ["&", "-", ".", "/", "://", ";", "=", "?", "^", "_"]:
                    if line.startswith(prefix):
                        self.harmful_url_segments.add(line)
                        break
    def is_harmful_css_id(self, css_id):
        return css_id in self.harmful_css_ids
    def is_harmful_css_classes(self, css_classes):
        for css_class in css_classes:
            if css_class in self.harmful_css_classess:
                return True
        return False
    def is_harmful_url(self, url):
        # warning: might be slow
        for harmful_url_segment in harmful_url_segments:
            if harmful_url_segment in url:
                return True
        return False
