import PyPDF2
import spacy
import os
import json


filenames = os.listdir('data-2017-05')


AD_MAP = [1, 1, 2, 1, 1, 1, 1, 2, 2]
AD_ENT_NAME = ['ID', 'Text', 'Landing Page', 'Targeting', 'Impressions', 'Clicks', 'Spend', 'Creation Date', 'End Date']
Entities = []

succsessfully_parsed = 0
overhaul_pages = len(filenames)

for name in filenames:
    file = PyPDF2.PdfFileReader('./data-2017-05/' + name)
    file_corpus = file.getPage(0).extractText()
    ad_index = -1
    entities = {'ID': '', 'Text': '', 'Landing Page': '', 'Targeting': '', 'Impressions': '', 'Clicks': '', 'Spend': '',
                'Creation Date': '', 'End Date': ''}
    counter = 0
    age_count = 0
    int_flag = False
    age = ''
    interests = ''
    lines = file_corpus.split('\n')
    for line in lines:
        if line == 'Ad ':
            ad_index += 1
            counter = 0
            continue
        if AD_MAP[ad_index] > counter:
            counter += 1
            continue
        entities[AD_ENT_NAME[ad_index]] += line
    if entities['ID'] != '' and entities['Text'] != '' and entities['Spend'] != '' and entities['Creation Date'] != ''\
            and entities['End Date'] != '':
        Entities.append(entities)
        succsessfully_parsed += 1

print("Succsessfully parsed: {}".format(succsessfully_parsed/overhaul_pages))

with open('data-2017-05' + '.json', 'w') as f:
    json.dump(Entities, f)
