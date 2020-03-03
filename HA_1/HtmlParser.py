import gzip
import json
import requests
import csv
import subprocess
import os
import re
from os.path import isfile
from io import BytesIO
import warnings
from io import StringIO
from io import TextIOWrapper
import numpy as np
import multiprocessing
from bs4 import BeautifulSoup
import lxml
from time import sleep


# -------------------------------------------------------------------
# Worker function of parallel pool for common crawl index analization
# -------------------------------------------------------------------
def get_links(CdxApi, IndexNum, Target_domain):
    print("Processing {0}".format(IndexNum))

    Req = requests.get(CdxApi + '?url=' + Target_domain + '&matchType=domain&output=json')
    if Req.status_code == 200:
        records = Req.content.decode()
        records = records.splitlines()
        length = list(range(len(records)))
        for i in length:
            records[i] = json.loads(records[i])
    return records



# -------------------------------------------------------------------
# Worker function of parallel pool for simultaneous page download
# -------------------------------------------------------------------
# I have met different dificulties. Still not done.
#
def get_page(record):
    opr_thr = {'processed': 0, 'downloaded': 0, 'bad_status': 0, 'duplicates_found': 0, 'enc_data_err': 0,
               'main_page_occurrence': 0}
    name = './raw_html/' + record['digest'] + '.txt'
    response = ""
    # status_failed = np.array(list(range(400, 418)) + list(range(500, 506)))
    if re.fullmatch(r'com,kotaku[)]/', record['urlkey']):
        opr_thr['main_page_occurrence'] = 1
        return opr_thr
    if record['status'] == '200':
        # We'll get the file via HTTPS so we don't need to worry about S3 credentials
        # Getting the file on S3 is equivalent however - you can request a Range
        prefix = 'https://commoncrawl.s3.amazonaws.com/'
        # We can then use the Range header to ask for just this set of bytes
        offset, length = int(record['offset']), int(record['length'])
        offset_end = offset + length - 1
        resp = requests.get(prefix + record['filename'], headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})
        # The page is stored compressed (gzip) to save space
        # We can extract it using the GZIP library
        raw_data = BytesIO(resp.content)
        f = gzip.GzipFile(fileobj=raw_data)
        # What we have now is just the WARC response, formatted:
        data = f.read().decode(record['charset'])
        if len(data):
            try:
                warc, header, response = data.strip().split('\r\n\r\n', 2)
                f.rewind()
                f.seek(len((warc + '\r\n\r\n').encode(record['charset'])) +
                       len((header + '\r\n\r\n').encode(record['charset'])))
                response = f.read()
                if not (isfile(name)):
                    try:
                        file = open(name, 'x')
                        souptest = BeautifulSoup(response, 'lxml')
                        # '<div class="post-content entry-content js_entry-content">.*</div>'
                        head = souptest.find('head')
                        body = souptest.find('body')
                        title = head.find('title')
                        author = head.find('meta', {'name': 'author'})
                        file.write(title.string + '\n')
                        file.write(("Author: %s" % author['content']) + '\n')
                        post = body.find('div', {'class': 'post-content'})
                        for member in post.contents:
                            if member.find('div', class_=['js_img-wrapper', 'img-wrapper', 'lazy-image', 'lazy-gif']):
                                file.write('<image_or_gif>\n')
                            if member.name == 'p' and member.attrs == {}:
                                file.write(member.text + '\n')
                            if member.name == 'p' and np.any('has-video' == np.array(member.attrs.get('class'))):
                                file.write('<gif_or_video>' + '\n')
                            if member.name == 'h4':
                                file.write(member.text + '\n')
                        file.close()
                        opr_thr['downloaded'] = 1
                    except:
                        opr_thr['enc_data_err'] = 1
                else:
                    opr_thr['duplicates_found'] = 1
            except:
                opr_thr['enc_data_err'] = 1
    else:
        opr_thr['bad_status'] = 1
    opr_thr['processed'] = 1
    return opr_thr


def print_proc_info(opr_thr):
    global opr
    opr['processed'] += opr_thr['processed']
    opr['downloaded'] += opr_thr['downloaded']
    opr['bad_status'] += opr_thr['bad_status']
    opr['duplicates_found'] += opr_thr['duplicates_found']
    opr['enc_data_err'] += opr_thr['enc_data_err']
    opr['main_page_occurrence'] += opr_thr['main_page_occurrence']
    print('| pr: %i / ' % opr['processed'] + '%i | ' % opr['to_download'] +
          'dl: %i | ' % opr['downloaded'] +
          'df: %i | ' % opr['duplicates_found'] +
          'dedc: %i | ' % opr['enc_data_err'] +
          'bs: %i |' % opr['bad_status'] +
          'mpo: %i |' % opr['main_page_occurrence'], end='\r')


# -------------------------------------------------------------
# Upload json index of common crawl and search specified domain
# -------------------------------------------------------------
def search_domain(domain):
    counter = 0
    IndexURL = "https://index.commoncrawl.org/collinfo.json"
    Data = requests.get(IndexURL).text
    Indexes = json.loads(Data)
    threads = []
    output = open("wanted_urls.csv", "w")
    keys = ['urlkey', 'timestamp', 'url', 'languages', 'mime', 'filename', 'length', 'charset', 'offset',
            'mime-detected', 'digest', 'status']
    writer = csv.DictWriter(output, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    print("You have %s CPUs...\nLet's use them all!" % multiprocessing.cpu_count())
    Pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for res in Indexes:
        proc = Pool.apply_async(get_links, (res['cdx-api'], (res['id']), domain))
        threads.append(proc)
    for proc in threads:
        records = proc.get()
        counter += len(records)
        writer.writerows(records)
        print("[*] Added %d results." % len(records))
        print("[*] Found a total of %d hits." % counter)
    Pool.close()
    output.close()


# ---------------------------
# Downloads full page
# ---------------------------
def download_pages(records, opr):
    print("\n\n\nDownload in process...")
    print("You have %s CPUs...\nLet's use them all!" % multiprocessing.cpu_count())
    print("\n\nKeywords:")
    print("proceeded - pr \ndownloaded - dl\nduplicates found - df\ndecoder error or data corruption - dedc\n"
          "bad response status - bs\nmain page occurrence - mpo\n")
    threads = []
    size = list(range(opr['to_download']))
    Pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()*10)
    for record, i in zip(records, size):
        proc = Pool.apply_async(get_page, (record,), callback=print_proc_info)
        threads.append(proc)
        if i % 10000 == 0:
            proc.wait()
    for proc in threads:
        proc.get()
    Pool.close()
    print('\n')


def bufcount(filename):
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])


def main():

    domain = 'https://kotaku.com/'
    # domain = input('Enter the main page of interested site:\n')

    if os.path.isfile("wanted_urls.csv"):
        char = 'a'
        while char != 'Y' and char != 'N' and char != 'y' and char != 'n':
            char = input("Do you want to update url index? (Y|N): ")
        if char == 'Y' or char == 'y':
            search_domain(domain)
    else:
        search_domain(domain)

    sleep(4)
    print('In parsing I use np string array comparision, It has Future Warning'
          'Now I suppress it, nut the code may not work in the future!!!')
    wanted_sites = open('wanted_urls.csv', 'r', newline='')
    # opr['to_download'] = wccount('wanted_urls.csv')
    opr['to_download'] = bufcount('wanted_urls.csv')
    reader = csv.DictReader(wanted_sites)
    download_pages(reader, opr)


if __name__ == '__main__':
    opr = {'to_download': 0, 'processed': 0, 'downloaded': 0, 'bad_status': 0, 'duplicates_found': 0,
           'enc_data_err': 0, 'main_page_occurrence': 0}
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
