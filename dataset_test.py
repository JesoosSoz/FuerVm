import os, sys
import spacy
from spacy.pipeline import Sentencizer

sentencizer = Sentencizer()
dir = "C:/Users/ViktorG/Desktop/Machine Learning/Datenset/cnn_stories_tokenized"
new_dir = "C:/Users/ViktorG/Desktop/Machine Learning/Datenset/cnn_text/"
new_dir2 = "C:/Users/ViktorG/Desktop/Machine Learning/Datenset/cnn_highlights/"
dir = dir + "/"

#spacy
nlp = spacy.load("en_core_web_sm")

soos = " "
nlp.add_pipe(sentencizer, first=True)

mem = os.listdir(dir)

for i in mem:
    try:
        old = open( dir + i, "r")
        newt = open(new_dir + "t-" + i + ".txt", "w+")
        newh = open(new_dir2 + "h-" + i + ".txt", "w+")
        text = old.read()

        body_list = nlp(text)
        body_list = [c.string.strip() for c in body_list.sents]

        list_index = 0

        while(list_index < len(body_list)):
            if("@highlight" in body_list[list_index]):
                text = body_list[:list_index - 1]
                
                highlight = body_list[list_index]
                highlight = highlight.replace("\n\n@highlight\n\n", ". ")
                highlight = highlight.replace("@highlight\n\n", "")

                list_index = len(body_list)
                newh.write(highlight)
            else:
                list_index = list_index + 1

        newt.write(soos.join(text))
        old.close()
        newt.close()
        newh.close()
    except:
        old.close()
        newt.close()
        newh.close()
