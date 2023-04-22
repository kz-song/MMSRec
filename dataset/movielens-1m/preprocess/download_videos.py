import sys
sys.path.append("../../../")
import os
import time
import yt_dlp
import argparse
from tqdm import tqdm
from general import load_meta_file
from youtube_search import YoutubeSearch
from utils.data_utils import audio_extract
from utils.basic_utils import check_dirs
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../raw', type=str)
    parser.add_argument('--processed_path', default='../processed', type=str)
    parser.add_argument('--meta_file', default='ml-1m/movies.dat', type=str)

    parser.add_argument('--video_outpath', default='videos', type=str, help='videos save path')
    parser.add_argument('--audio_outpath', default='audios', type=str, help='audios save path')
    parser.add_argument('--text_outpath', default='texts', type=str, help='texts save path')
    args = parser.parse_args()
    return args


class MovieLensDownload(object):
    def __init__(self,
                 args,
                 metas):
        self.metas = metas

        self.raw_path = args.raw_path
        self.processed_path = args.processed_path

        self.video_outpath = args.video_outpath
        self.audio_outpath = args.audio_outpath
        self.text_outpath = args.text_outpath

    def search_video(self, query):
        try:
            query = query + " trailer"
            video_infos = YoutubeSearch(query, max_results=3).to_dict()
            if not len(video_infos):
                return None
            video_infos = video_infos[:3]
            video_urls = ["https://www.youtube.com" + video_info["url_suffix"] for video_info in video_infos]
            return video_urls
        except:
            return None

    def write_text(self, meta, path):
        text = f"name: {meta['name']}, tag: {meta['tag']}"
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

    def download_video(self, id, meta, vurls):
        vision_path = os.path.join(self.processed_path, self.video_outpath, f'{id}.mp4')
        audio_path = os.path.join(self.processed_path, self.audio_outpath, f'{id}.wav')
        text_path = os.path.join(self.processed_path, self.text_outpath, f'{id}.txt')

        ydl_opts = {
            'writeinfojson': False,
            'outtmpl': vision_path,
            'merge_output_format': 'mp4',
            'format': 'best[height<=360]',
            'skip_download': False,
            'ignoreerrors': True,
            'no_warnings': True,
            'quiet': True
        }
        try:
            for vurl in vurls:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    if not os.path.exists(vision_path):
                        result = ydl.download([vurl])
                        if result != 0:
                            continue
                    if not os.path.exists(audio_path):
                        audio_extract(vision_path, audio_path)
                    if not os.path.exists(text_path):
                        self.write_text(meta, text_path)
                    return True
            print(f'Fail to download {id} : {meta["name"]} : {vurls[0]}')
            return False
        except:
            print(f'Fail to download {id} : {meta["name"]} : {vurls[0]}')
            return False

    def process_one_item(self, id, meta):
        video_urls = self.search_video(meta["name"])
        if video_urls is None:
            print(f'Fail to search {id} : {meta["name"]}')
            return id, meta, False
        state = self.download_video(id, meta, video_urls)
        return id, meta, state

    def process(self):
        loop = 0
        while loop < 10:
            print(f'Download Loop {loop}')
            failed_items = {}
            with ThreadPoolExecutor(max_workers=max(8 // (loop + 1), 1)) as executor:
                process_list = []

                for id, meta in self.metas.items():
                    process = executor.submit(self.process_one_item, id, meta)
                    process_list.append(process)

                progress_bar = tqdm(range(len(process_list)))
                for process in as_completed(process_list):
                    id, meta, state = process.result()
                    if state is False:
                        failed_items[id] = meta
                    progress_bar.set_postfix_str(f"fail {len(failed_items)}")
                    progress_bar.update(1)

            print(f"Success: {len(self.metas) - len(failed_items)}, fail: {len(failed_items)}")
            self.metas = failed_items
            if len(self.metas) == 0:
                break
            time.sleep(10)
            loop += 1
        print("Download Finish!")


if __name__ == "__main__":
    args = parse_args()
    check_dirs(os.path.join(args.processed_path, args.video_outpath))
    check_dirs(os.path.join(args.processed_path, args.audio_outpath))
    check_dirs(os.path.join(args.processed_path, args.text_outpath))

    metas = load_meta_file(os.path.join(args.raw_path, args.meta_file))

    api = MovieLensDownload(args, metas)
    api.process()


