from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import pandas as pd
import datetime
import csv
import re 
import os

#api_key = os.getenv('GOOGLE_API_KEY')
#youtube = build('youtube', 'v3', developerKey = api_key)

def extract_instructor_from_description(description):
    """
    Description : Extracts the instructor from the playlist's description.

    Inputs:
    - description : str - Playlist's description

    Outputs:
    - instructor : str - Instructor's name extracted from the description
    """
    pattern = r"Instructor(s)?: (.+)"
    match = re.search(pattern, description)
    if match:
        instructor = match.group(2)
        return instructor
    else:
        return None
    
def get_video_title(video_id):
    """
    Description : Gets the title of a video from YouTube from its id.

    Inputs:
        - video_id: YouTube video's id.

    Outputs:
        - title : The YouTube video's title.

    """
    api_key = os.getenv('GOOGLE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_info = youtube.videos().list(part='snippet', id=video_id).execute()
    title = video_info['items'][0]['snippet']['title']
    return title

def series_gen(csv_file, sid, eid, cid, save = True):
    """
    Description : Generates data for series from a csv file containing playlists' information.

    Inputs:
    - csv_file : str - Path to csv file containing playlists' information
    - sid : int - Initial value for the serie's id
    - eid : int - Initial value for the episode's id
    - cid : int - Initial value for the chapter's id
    - save : bool - Save the data generated, default value = True

    Sortie:
    - series : list - List of the generated series' data
    """

    if sid == -1:
        sid_iter = 0
    else:
        sid_iter = sid
        
    if eid == -1:
        eid_iter = 0
    else:
        eid_iter = eid
    
    if cid == -1:
        cid_iter = 0
    else:
        cid_iter = cid

    series = []
    episodes = []
    chapters = []

    api_key = os.getenv('GOOGLE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=api_key)

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            playlist_id, channel_name, category = row
            print(f'Extracting subtitles from playlist n° {sid_iter}\t with id {playlist_id}')
            playlist_info = youtube.playlists().list(part="snippet", id=playlist_id).execute()
            playlist_title = playlist_info['items'][0]['snippet']['title']
            playlist_response = youtube.playlists().list(
                part='snippet',
                id=playlist_id
            ).execute()
       
            created_at = playlist_response['items'][0]['snippet']['publishedAt']
            title = playlist_response['items'][0]['snippet']['title']
            description = playlist_response['items'][0]['snippet']['description'].replace('\n', ' ').replace('\r', '')
            instructor = extract_instructor_from_description(description)
        

            videos = []

            next_page_token = None
            while True:
                res = youtube.playlistItems().list(
                    playlistId=playlist_id,
                    part='snippet',
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                videos += res['items']
                next_page_token = res.get('nextPageToken')
                if next_page_token is None:
                    break

            playlist_data = {
                'Sid': sid_iter,
                'Category': category,
                'Course Number': playlist_id,
                'Serie Name': title.replace('|',','), 
                'Professor Name': instructor,
                'Date':created_at,  
                'Url': f'https://www.youtube.com/playlist?list={playlist_id}',
                'Corpus': channel_name,
                'Description': description if description else '',
            }

            series.append(playlist_data)   
            
            eid_final = len(episodes) + eid_iter

            episodes_gen(channel_name, episodes, chapters, playlist_id, sid_iter, eid, eid_final, cid_iter, save=True)

            sid_iter += 1

    df = pd.DataFrame(series)
    if save:
        df.to_csv('series.csv', index = False, sep = '|')


def episodes_gen(channel_name, episodes, chapters, playlist_id, sid, eid_final, cid, save = True):
    """
    Description : Generates data for episodes from a playlist.

    Inputs:
    - channel_name : str - Channel name
    - episodes : list - Generated episodes' list
    - chapters : list - Generated chapters' list
    - playlist_id : str - Playlist's id
    - sid : int - Serie's id
    - eid : int - Initial value for the episode's id
    - eid_final : int - Final value for the episode's id
    - cid : int - Initial value for the chapter's id
    - save : bool - Save the data generated, default value = True

    Sortie:
    - length : int - Total number of the generated episodes
    """

    video_ids = []
    episode_names = []
    number = 0
    n = number + 1
    api_key = os.getenv('GOOGLE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=api_key)

    try:
        playlist = youtube.playlists().list(
            part='snippet',
            id=playlist_id
        ).execute()['items'][0]['snippet']

        try:
            playlist_items = []
            next_page_token = None
            while True:
                res = youtube.playlistItems().list(
                    playlistId=playlist_id,
                    part='contentDetails, snippet',
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                playlist_items += res['items']
                next_page_token = res.get('nextPageToken')
                if not next_page_token:
                    break

            for item in playlist_items:
                print(f"Extracting subtitles for whole video n° {item['contentDetails']['videoId']}")
                video_ids.append(item['contentDetails']['videoId'])
                episode_names.append(item['snippet']['title'])

        except HttpError as e:
            print(f'Error retrieving playlist information for playlist {playlist_id}. Exception type: {type(e)}. Exception message: {str(e)}')

        for video_id, episode_name in zip(video_ids, episode_names):
            video_response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            if 'items' in video_response and len(video_response['items']) > 0:
                published_at = video_response['items'][0]['snippet'].get('publishedAt')
            else:
                published_at = None

            playlist_info = {
                'Eid': eid_final,
                'Sid': sid,
                'Course Number': playlist_id,
                'Number In Series': f'Lecture{n}',
                'N In Series': number,
                'Title': playlist['title'].replace('|', ','),
                'Date': published_at,
                'Url': f'https://www.youtube.com/playlist?list={playlist_id}',
                'Ressources': f'https://www.youtube.com/playlist?list={playlist_id}',
                'Description': playlist['description'].replace('\n', ' ').replace('\n', ' ').replace('\r', ''),
                'Video': video_id,
                'Corpus': channel_name
            }
            episodes.append(playlist_info)
            cid_final = len(chapters) + cid

            chapters_gen(channel_name, chapters, video_id, episode_name, sid, eid_final, cid, cid_final, save=True)

            eid_final += 1
            number += 1
            n += 1

        df1 = pd.DataFrame(episodes)

    except HttpError as e:
        print(f'Error retrieving playlist information for playlist {playlist_id}. Exception type: {type(e)}. Exception message: {str(e)}')
        return None

    if save:
        df1.to_csv('episodes.csv', index = False, sep = '|')

    return len(episodes)


def chapters_gen(channel_name, chapters, video_id, sid, eid, cid, cid_final, save = True):
    """
    Description : Generates data for chapters from a YouTube video.

    Inputs:
        - channel_name: str - Channel name
        - chapters: list - List of existing chapters
        - video_id: str - YouTube video's id
        - episode_name: str - Episode name
        - sid: int - Serie's id
        - eid: int - Episode's id
        - cid: int - Initial value for the chapter's id
        - cid_final: int - Final value for the chapter's id
        - save : bool - Save the data generated, default value = True

    Outputs:
        - Number of chapters generated

    Fonctionnement:
        - Récupère le transcript de la vidéo YouTube.
        - Divise le transcript en segments d'une durée minimale spécifiée.
        - Génère les informations des chapitres pour chaque segment.
        - Ajoute les informations des chapitres à la liste existante.
        - Incrémente les ID de chapitres.
        - Enregistre les chapitres dans un fichier CSV si l'indicateur 'save' est activé.
    """
    cid_start = cid
    segments = []
    current_segment = ''
    current_duration = 0
    current_start = 0
    current_end = 0
    min_segment_duration = 20 * 60

    unwanted_words = ['\n', '\t', '\r', '|', '[PROFESSOR]', '[Voiceover]', '[AUDIENCE]', '[APPLAUSE]', '[INAUDIBLE]']

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
        total_duration_seconds = transcript[-1]['start'] + transcript[-1]['duration']

        if transcript and total_duration_seconds > 3600:
            print(f'Extracting subtitles for video n° {cid_final} and dividing it')
            # Divide the transcript into segments of a specified minimal duration
            for line in transcript:
                text = line['text']
                duration = line['duration']
                start = line['start']
                end = start + duration

                if current_duration + duration > min_segment_duration:
                    current_segment += text + ' '
                    current_duration += duration

                    if text.endswith(('.', '!', '?')):
                        # Add the segment to the segments' list
                        segments.append((current_segment.strip(), current_start, current_end))
                        current_segment = ''
                        current_duration = 0
                        current_start = 0
                        current_end = 0

                else:
                    if current_duration == 0:
                        current_start = start

                    current_segment += text + ' '
                    current_duration += duration
                    current_end = end

            title_number = 0
            num_parts = len(segments)
            remaining_duration = total_duration_seconds - segments[-1][1]

            video_title = get_video_title(video_id)

            # Generate the information of chapters for every segment
            for i, part in enumerate(segments):
                if i < num_parts - 1:
                    segment_text, segment_start, segment_end = part
                else:
                    segment_text, segment_start, _ = part
                    segment_end = segment_start + remaining_duration

                for word in unwanted_words:
                    segment_text = segment_text.replace(word, ' ')

                start_time = str(datetime.timedelta(seconds=segment_start)).split('.')[0]
                end_time = str(datetime.timedelta(seconds=segment_end)).split('.')[0]

                # Add chapter information to the chapters' list
                chapter = {
                    'Cid': cid_final,
                    'Eid': eid,
                    'Sid': sid,
                    'Title': f"{video_title} Part {title_number}".replace('|', ','),
                    'Text': segment_text,
                    'BeginTimestamp': start_time,
                    'EndTimestamp': end_time,
                    'Corpus': channel_name,
                }
                chapters.append(chapter)
                title_number += 1
                cid_final += 1

        else:
            print(f'Extracting subtitles for video n° {cid_final} without dividing it')
            try:
                text = ' '.join([line['text'] for line in transcript])
                cleaned_transcript_text = text
                for word in unwanted_words:
                    cleaned_transcript_text = cleaned_transcript_text.replace(word, ' ')

                if not cleaned_transcript_text.endswith(('.', '!', '?')):
                    cleaned_transcript_text += '.'

                title = get_video_title(video_id)
                chapters_info = {
                    'Cid': cid_final,
                    'Eid': eid,
                    'Sid': sid,
                    'Title': title.replace('|', ','),
                    'Text': cleaned_transcript_text,
                    'BeginTimestamp': str(datetime.timedelta(seconds=0)).split('.')[0],
                    'EndTimestamp': str(datetime.timedelta(seconds=total_duration_seconds)).split('.')[0],
                    'Corpus': channel_name,
                }
                chapters.append(chapters_info)
                cid_final += 1
            except Exception as e:
                print(f"Error retrieving transcript for video {video_id}. Exception type: {type(e)}. Exception message: {str(e)}")

    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}. Exception type: {type(e)}. Exception message: {str(e)}")

    num_rows = len(chapters) - cid_start
    if len(chapters) != num_rows:
        print(f"Warning: Number of rows in the 'chapters' file does not match the expected count. Expected: {num_rows}, Actual: {len(chapters)}")

    df2 = pd.DataFrame(chapters)
    if save:
        df2.to_csv('chapters.csv', index = False, sep = '|')
    return len(chapters)
