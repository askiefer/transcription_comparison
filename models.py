import argparse
import os
import time
import utils
import csv
import pandas as pd
import datetime

from common import get_texts, CSV_FILEPATH


NUM_VIDEOS = 20
DIRPATH = "/Users/annas.kiefer/Desktop/transcribe"
FILEPATH = f"{DIRPATH}/DD_transcripts_serota.csv"


def run_deepgram_model(video_filepath: str, video_id: str, start_time: int, end_time: int):

    filepath_out = f"{DIRPATH}/deepgram/{video_id}_deepgram.txt"


def run_assemblyai_model(video_filepath: str, video_id: str, start_time: int, end_time: int):

    filepath_out = f"{DIRPATH}/assembly/{video_id}_assembly.txt"

    api_key = os.getenv("AAI_API_KEY")
    if api_key is None:
        raise RuntimeError("AAI_API_KEY environment variable not set. Try setting it now, or passing in your "
                           "API key as a command line argument with `--api_key`.")

    # Create header with authorization along with content-type
    header = {
        'authorization': api_key,
        'content-type': 'application/json'
    }

    # don't need to upload file
    # upload_url = utils.upload_file(video_filepath, header)

    start_run_time = time.time()
    # Request a transcription
    transcript_response = utils.request_transcript(video_filepath, header, start_time, end_time)

    # Create a polling endpoint that will let us check when the transcription is complete
    polling_endpoint = utils.make_polling_endpoint(transcript_response)
    print(f"{video_id}", polling_endpoint)

    # Wait until the transcription is complete
    utils.wait_for_completion(polling_endpoint, header)
    end_run_time = time.time()

    # Request the paragraphs of the transcript
    # paragraphs = utils.get_paragraphs(polling_endpoint, header)
    response = utils.get_sentences(polling_endpoint, header)

    print(f"{video_id} Transcription time: {(end_run_time - start_run_time)}")

    # Save and print transcript
    with open(filepath_out, 'w') as f:
        for r in response:
            f.write(r['text'] + '\n')

    return


def sort_videos(videos):
    videos_with_full_time = []
    for vid_id, info in videos.items():
        vid_start_time = info["start_time"]
        vid_end_time = info["end_time"]
        full_time = vid_end_time - vid_start_time
        info["full_time"] = full_time
        info["video_id"] = vid_id
        videos_with_full_time.append(info)

    vids = sorted(videos_with_full_time, key=lambda x: x["full_time"], reverse=True)
    return vids

def main():

    with open(CSV_FILEPATH) as f:
        reader = csv.DictReader(f)

        videos = {}

        for row in reader:
            hearing_date = row["Hearing date"]

            if hearing_date[-2:] == "18":  # only get videos from 2018 hearing
                video_link = row["Link to video"]
                video_id = os.path.basename(video_link).split(".mp4")[0]
                full_video_link = video_link.split("#t=")[0]
                if video_id not in videos:
                    start_time = int(row["Start"])
                    videos[video_id] = {"video_link": full_video_link, "start_time": start_time}
                end_time = int(row["End"])
                videos[video_id]["end_time"] = end_time

        videos = sort_videos(videos)

        video_count = 0
        for info in videos:
            if video_count <= NUM_VIDEOS:
                vid_link = info["video_link"]
                # convert end_time from s to ms
                vid_start_time = info["start_time"] * 1000
                vid_end_time = info["end_time"] * 1000
                vid_id = info["video_id"]
                full_time = info["full_time"]
                run_deepgram_model(vid_link, vid_id, vid_start_time, vid_end_time)
                # run_assemblyai_model(vid_link, vid_id, vid_start_time, vid_end_time)
                video_count += 1
        #         break

if __name__ == '__main__':
    main()


