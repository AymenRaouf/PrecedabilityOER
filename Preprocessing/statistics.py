import pandas as pd
from datetime import datetime

def generate_series_statistics(csv_file):
    df = pd.read_csv(csv_file, sep = '|')
    
    total_series = len(df)
    
    #total_episodes = df['Episodes'].sum()   
    #average_episodes_per_series = total_episodes / total_series    
    series_per_channel = df['Corpus'].value_counts()    
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    series_per_year = df.groupby('Year')['Sid'].nunique()
    
    print(f"Toal number of series : {total_series}")
    #print(f"Nombre total d'épisodes : {total_episodes}")
    #print(f"Moyenne du nombre d'épisodes par série : {average_episodes_per_series}")
    print(f"Number of series per YouTube channel : ")
    print(series_per_channel.to_string())
    print("Number of series per year :")
    print(series_per_year.to_string(header=False))



def generate_episode_statistics(csv_file):
    df = pd.read_csv(csv_file, sep = '|')
    
    total_episodes = len(df)

    episodes_per_year = df.groupby(pd.to_datetime(df['Date']).dt.year)['Eid'].count()
    episodes_per_channel = df['Resources'].value_counts()
    average_episodes_per_series = total_episodes / df['Sid'].nunique()
    
    print(f"Total number of episodes : {total_episodes}")
    print(f"Moyenne du nombre d'épisodes par série : {average_episodes_per_series}")
    print("Nombre d'épisodes par an :")
    print(episodes_per_year.to_string())
    print("Nombre d'épisodes par chaîne YouTube :")
    print(episodes_per_channel.to_string())


def generate_chapter_statistics(csv_file):
    df = pd.read_csv(csv_file, sep = '|')
    
    total_chapters = len(df)
    
    average_chapters_per_episode = total_chapters / df['Eid'].nunique()
    average_chapters_per_series = total_chapters / df['Sid'].nunique()
    average_words_per_chapter = df['Text'].apply(lambda x: len(x.split())).mean()
    max_words_per_chapter = df['Text'].apply(lambda x: len(x.split())).max()
    min_words_per_chapter = df['Text'].apply(lambda x: len(x.split())).min()
    chapters_per_channel = df['Corpus'].value_counts()
    
    durations = pd.to_datetime(df['EndTimestamp']) - pd.to_datetime(df['BeginTimestamp'])
    average_duration_per_chapter =str(durations.mean()).split('.')[0]
    
    print(f"Nombre total de chapitres : {total_chapters}")
    print(f"Nombre moyen de chapitres par épisode : {average_chapters_per_episode}")
    print(f"Nombre moyen de chapitres par séries : {average_chapters_per_series}")
    print(f"Nombre moyen de mots par chapitre : {average_words_per_chapter}")
    print(f"Nombre max de mots par chapitre : {max_words_per_chapter}")
    print(f"Nombre min de mots par chapitre : {min_words_per_chapter}")
    print(f"Durée moyenne d'un chapitre : {average_duration_per_chapter}")
    print("Nombre de chapitres par chaîne YouTube :")
    print(chapters_per_channel.to_string())

def extract_year(date_str):
    date_str = date_str.replace('Spring ', '').replace('Fall ', '')
    if len(date_str) != 4:
        date_str = date_str[:4]
    return(date_str)


def generate_statistics(series, episodes, chapters):
    series_data = []
    episodes_data = []
    chapters_data = []
    chapters_per_category_data = []
    episodes_per_category_data = []

    df_series = pd.read_csv(series, sep = '|')
    total_series = len(df_series)
    series_per_channel = df_series['Corpus'].value_counts()
    df_series['Date'] = df_series['Date'].apply(lambda x: extract_year(x))
    series_per_year = df_series.groupby(df_series['Date'])['Sid'].nunique()

    series_data = {
        'Total Series': total_series,
        'Series per Channel': series_per_channel,
        'Series per Year': series_per_year
    }

    df_episodes = pd.read_csv(episodes, sep = '|')
    total_episodes = len(df_episodes)
    episodes_per_channel = df_episodes['Corpus'].value_counts()
    average_episodes_per_series = total_episodes / df_episodes['Sid'].nunique()

    episodes_data = {
        'Total Episodes': total_episodes,
        'Episodes per Channel': episodes_per_channel,
        'Average Episodes per Series': average_episodes_per_series
    }
    merged_data = pd.merge(df_episodes, df_series, on='Sid')
    episodes_per_category_data = merged_data.groupby('Category')['Eid'].count()


    df_chapters = pd.read_csv(chapters, sep = '|')
    total_chapters = len(df_chapters)
    average_chapters_per_episode = total_chapters / df_chapters['Eid'].nunique()
    average_chapters_per_series = total_chapters / df_chapters['Sid'].nunique()
    average_words_per_chapter = df_chapters['Text'].apply(lambda x: len(x.split())).mean()
    chapters_per_channel = df_chapters['Corpus'].value_counts()

    chapters_data = {
        'Total Chapters': total_chapters,
        'Average Chapters per Episode': average_chapters_per_episode,
        'Average Chapters per Series': average_chapters_per_series,
        'Average Words per Chapter': average_words_per_chapter,
        'Chapters per Channel': chapters_per_channel
    }

    # Join chapters and series data on 'Sid' column
    merged_data = pd.merge(df_chapters, df_series, on='Sid')
    chapters_per_category_data = merged_data.groupby('Category')['Cid'].count()

   
            
            
    return series_data, episodes_data, episodes_per_category_data, chapters_data, chapters_per_category_data