# -*- coding:utf8 -
import requests
import re
import os, sys
import json
import random
from bs4 import BeautifulSoup
import threading
import time
import urllib

user_agents = ['Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
               'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
               'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11']


class get_qmkg(object):
    def __init__(self, address, uid):
        self.ur = 'https://kg.qq.com/node/personal?uid=' + uid
        self.url = 'http://node.kg.qq.com/play?s='
        self.listurl = 'http://node.kg.qq.com/cgi/fcgi-bin/kg_ugc_get_homepage?jsonpCallback=callback_0&type=get_ugc&num=8&share_uid=' + str(
            uid) + '&start='
        self.audionames = []
        self.raw_urls = []
        self.audiourls = []
        self.address = address
        self.points = []

    def get_html(self, url):
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        else:
            print(url + 'request failed')
            # notFailed = False
            return None

    def parse_response(self, response):
        rtext = response.text
        soup = BeautifulSoup(rtext, 'html.parser')
        lis = soup.find_all('li', class_="mod_playlist__item")
        for item in lis:
            self.raw_urls.append(self.url + item.get("data-shareid"))
            title = item.find('a', class_="mod_playlist__work").get_text()
            self.audionames.append(title)

    def parse_stream(self):
        for one_url in self.raw_urls:
            html = self.get_html(one_url).text
            pattern = 'playurl":"(.*?)","playurl_video'
            url = re.findall(pattern, html)[0]
            self.audiourls.append(url)

            pattern = '"score":(.*?),"scoreRank"'
            point = re.findall(pattern, html)[0]
            self.points.append(point)

            pattern = '"uid":(.*?)},'
            newuid = re.findall(pattern, html)[0]
            newuid = newuid.split(':')[-1].replace('"', '')
            if newuid not in useduid and newuid not in uidlist and len(newuid) > 13:
                uidlist.append(newuid)

    def file_path(self):
        # todo: 改变qmkg歌曲对应的qq原唱的歌名as you like
        for num in range(0, len(self.audiourls)):
            if '主打' in self.audionames[num]:
                self.audionames[num] = self.audionames[num][2:]
            if int(self.points[num]) < 500:
                continue

            songlist.append(self.audionames[num])  # 爬取的用户歌曲名称，添加到全局的list里

            self.audionames[num] += '--' + self.points[num]
            path = os.path.join(self.address, self.audionames[num])
            with open(path + '.mp3', 'wb') as f:
                this_url = self.audiourls[num]
                if len(this_url) == 0 or this_url[0] == ' ':
                    continue
                content = self.get_html(this_url).content
                f.write(content)
                sys.stdout.write('QMKG...:' + self.audionames[num] + '\n')
                sys.stdout.flush()

    def main(self):
        response = self.get_html(self.ur)
        self.parse_response(response)
        self.parse_stream()
        self.file_path()


class get_qqMusic(object):
    def __init__(self, address):
        self.address = address

    def get_song_list(self, keyword):
        keyword = urllib.parse.quote(keyword)
        url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp?aggr=1&cr=1&p=1&n=20&w=%s' % keyword
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url, headers=headers).text.encode('gbk', 'ignore').decode('gbk').split('callback')[-1].strip('()')
        response = json.loads(response)
        return response['data']['song']

    def get_mp3_url(self, songs, num):
        # todo: arg num is changeable
        media_mid = songs['list'][num]['media_mid']
        url_1 = 'https://c.y.qq.com/base/fcgi-bin/fcg_music_express_mobile3.fcg?g_tk=5381&cid=205361747&songmid=%s&filename=C400%s.m4a&guid=6800588318' % (
            media_mid, media_mid)
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url_1, headers=headers).json()
        vkey = response['data']['items'][0]['vkey']
        if vkey:
            url_2 = 'http://dl.stream.qqmusic.qq.com/C400%s.m4a?vkey=%s&guid=6800588318&uin=0&fromtag=66' % (media_mid, vkey)
            return url_2
        return None

    def download_mp3(self, url, filename):
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url, headers=headers).content

        path = os.path.join(self.address, filename)
        with open(path + '.m4a', 'wb') as f:
            f.write(response)

    def main(self):
        time.sleep(5)
        while True:
            if len(songlist) == 0:
                time.sleep(2)
                continue
            time.sleep(1)
            name = songlist.pop(0)
            songs = self.get_song_list(name)
            if songs['totalnum'] == 0:
                continue
            else:
                url = self.get_mp3_url(songs, 0)
                if not url:
                    continue
                else:
                    songname = songs['list'][0]['songname']
                    self.download_mp3(url, songname)
                    print('QQmusic...' + songname)


def threadCrawler_qmkg(start_uid, path, thread_id):  # the last arg could be abbreviated
    while True:
        if len(songlist) > 50:
            time.sleep(2)
        time.sleep(1)
        # notFailed = True
        if start_uid in useduid:  # avoid repeated uid
            if len(uidlist) == 0:
                break
            start_uid = uidlist.pop(0)
            continue

        if len(start_uid) > 14:
            uidlist.append(start_uid[:-2])

        get_qmkg(path, start_uid).main()

        if len(uidlist) == 0:  # uid used out
            break
        useduid.append(start_uid)
        # print(uidlist)
        start_uid = uidlist.pop(0)
    # threadStatus[thread_id] = 1


def simpleCrawler_qqMusic(path):
    get_qqMusic(path).main()


if __name__ == '__main__':
    # threadStatus = [0]
    songlist = ['聪哥最帅']
    uidlist = ['619e99872224338330', '649d9482232c348f37', '66989c852329378934', '639b9a81272d3f',
               '63959c86202d3f8937', '609e9d8d232b338a32', '639a9f86242e358c35']
    useduid = []

    abspath = os.path.abspath('.')  # 获取绝对路径
    os.chdir(abspath)
    qmkg_path = os.path.join(abspath, 'qmkg')  # 放到qmkg目录下
    if not os.path.exists(qmkg_path):
        os.mkdir(qmkg_path)

    origin_path = os.path.join(abspath, 'origin')  # 放到origin目录下
    if not os.path.exists(origin_path):
        os.mkdir(origin_path)

    print('initializing...')
    threadNum = 7
    for i in range(threadNum):  # create crawler threads
        # threadStatus.append(0)
        t = threading.Thread(target=threadCrawler_qmkg, args=(uidlist[0], qmkg_path, i))
        t.daemon = True
        t.start()
        t = threading.Thread(target=simpleCrawler_qqMusic, args=(origin_path,))
        t.daemon = True
        t.start()
        print('Thread-pair-' + str(i) + ' initialized')
        # print(uidlist)
        uidlist.pop(0)

    while True:
        time.sleep(40)  # create more threads because the server may kill some of them every minute
        # threadStatus.append(0)
        t = threading.Thread(target=threadCrawler_qmkg, args=(uidlist[0], qmkg_path, i))
        t.daemon = True
        t.start()
        t = threading.Thread(target=simpleCrawler_qqMusic, args=(origin_path,))
        t.daemon = True
        t.start()
        print('new thread-pair added')
        uidlist.pop(0)
