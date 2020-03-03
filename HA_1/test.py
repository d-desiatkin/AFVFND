import os
import os.path
import jsonlines
import json
from bs4 import BeautifulSoup
import numpy as np
import lxml


# path = './test_parse/'
# objects_to_parse = os.listdir(path)
# for object in objects_to_parse:
#     file = open(path + object, 'rb')
#     souptest = BeautifulSoup(file, 'lxml')
#     # '<div class="post-content entry-content js_entry-content">.*</div>'
#     head = souptest.find('head')
#     body = souptest.find('body')
#     title = head.find('title')
#     author = head.find('meta', {'name': 'author'})
#     print(title.string)
#     print("Author: %s" % author['content'])
#     post = body.find('div', {'class': 'post-content'})
#     for member in post.contents:
#         if member.find('div', class_=['js_img-wrapper', 'img-wrapper', 'lazy-image', 'lazy-gif']):
#             print('<image_or_gif>')
#         if member.name == 'p' and member.attrs == {}:
#             print(member.text)
#         if member.name == 'p' and np.any('has-video' == np.array(member.attrs.get('class'))):
#             print('<gif_or_video>')
#         if member.name == 'h4':
#             print(member.text)
read_path = './raw_html/'
path = './text/'
js_out = 'output.txt'
writer = open(path + js_out, mode='w')
raw_text_files = os.listdir(read_path)
for obj in raw_text_files:
    file = open(read_path + obj, 'r')
    text = file.read()
    text = text + '\n<|endoftext|>'
    writer.write(text)
writer.close()
