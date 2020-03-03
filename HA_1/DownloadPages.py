import gzip
import json
import requests
import csv
import subprocess
import time
import sys
import re
import sqlite3
import os
from os.path import isfile
from io import BytesIO
from io import StringIO
from io import TextIOWrapper
import numpy as np
import multiprocessing
from bs4 import BeautifulSoup
from lxml import html


# -------------------------------------------------------------------
# Worker function of parallel pool for common crawl index analization
# -------------------------------------------------------------------
def get_links(CdxApi, IndexNum, Target_domain):
    print("Processing {0}".format(IndexNum))
    Req = requests.get(CdxApi + '?url=' + Target_domain + '&matchType=domain&output=json')
    if Req.status_code == 200:
        records = Req.content.decode()
        records = records.splitlines()
        for i in range(len(records)):
            records[i] = json.loads(records[i])
    return records


# -------------------------------------------------------------------
# Worker function of parallel pool for simultaneous page download
# -------------------------------------------------------------------
# I have met different dificulties. Still not done.
#
def get_page(record):
    opr_thr = {'processed': 0, 'downloaded': 0, 'bad_status': 0, 'duplicates_found': 0, 'enc_data_err': 0}
    name = './raw_html/' + record['digest'] + '.html'
    response = ""
    # status_failed = np.array(list(range(400, 418)) + list(range(500, 506)))
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
                        file = open(name, 'xb')
                        file.write(response)
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
    print('| pr: %i / ' % opr['processed'] + '%i | ' % opr['to_download'] +
          'dl: %i | ' % opr['downloaded'] +
          'df: %i | ' % opr['duplicates_found'] +
          'dedc: %i | ' % opr['enc_data_err'] +
          'bs: %i |' % opr['bad_status'], end='\r')


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
          "bad response status - bs\n")
    threads = []
    size = list(range(opr['to_download']))
    Pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for record, i in zip(records, size):
        proc = Pool.apply_async(get_page, (record,), callback=print_proc_info)
        threads.append(proc)
        if i % 10000 == 0:
            proc.wait()
    for proc in threads:
        proc.get()
    Pool.close()


def main():
    if os.path.isfile("wanted_urls.csv"):
        char = 'a'
        while char != 'Y' and char != 'N' and char != 'y' and char != 'n':
            char = input("Do you want to update url index? (Y|N): ")
        if char == 'Y' or char == 'y':
            search_domain('https://kotaku.com/')
    else:
        search_domain('https://kotaku.com/')

    wanted_sites = open('wanted_urls.csv', 'r', newline='')
    out = subprocess.Popen(['wc', '-l', 'wanted_urls.csv'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                           ).communicate()[0]
    opr['to_download'] = int(out.partition(b' ')[0])
    reader = csv.DictReader(wanted_sites)
    download_pages(reader, opr)


if __name__ == '__main__':
    opr = {'to_download': 0, 'processed': 0, 'downloaded': 0, 'bad_status': 0, 'duplicates_found': 0, 'enc_data_err': 0}

    main()
