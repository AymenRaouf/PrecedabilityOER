import pandas as pd
import re


def test_id_logicS(csv_file):
    df = pd.read_csv(csv_file, sep = '|')

    unique_sid = df['Sid'].nunique()
    total_sid = len(df)
    if unique_sid == total_sid :
        print("Test : Uniqueness of identifers sid - PASS")
        sid = True
    else :
        print("Test : Uniqueness of identifers sid - FAIL")
        sid = False
    
    return sid


def test_id_logicE(csv_file):     
    df = pd.read_csv(csv_file, sep = '|')
    unique_eid = df['Eid'].nunique()
    total_eid = len(df)
    if unique_eid == total_eid :
        print("Test : Uniqueness of identifers eid - PASS")
        eid = True
    else :
        print("Test : Uniqueness of identifers eid - FAIL")
        eid = False
    
    return eid


def test_id_logicC(csv_file):     
    df = pd.read_csv(csv_file, sep = '|')
    unique_cid = df['Cid'].nunique()
    total_cid = len(df)
    if unique_cid == total_cid :
        print("Test : Uniqueness of identifers cid - PASS")
        cid = True
    else :
        print("Test : Uniqueness of identifers cid - FAIL")
        cid = False
        
    return cid


def test_id_logic(series_csv, episodes_csv, chapters_csv):
    sid =  test_id_logicS(series_csv)
    eid = test_id_logicE(episodes_csv)
    cid = test_id_logicC(chapters_csv)

    is_logic = True
    
    while sid == False or eid == False or cid == False :
        is_logic = False
        print(f"Test : Logic of identifiers cid, eid and sid - FAIL (Row {i+1})")
        break
        
    if is_logic :
        print("Test : Logic of identifiers cid, eid and sid - PASS")


def test_transcripts(csv_file):
    df = pd.read_csv(csv_file, sep = '|')
    
    clean_transcript = True

    for i, row in df.iterrows():
        transcript = row['Text']

        cleaned_transcript = re.sub(r'[\t\n]', '', transcript)
        cleaned_transcript = cleaned_transcript.replace('[Voiceover]', '')

        last_character = cleaned_transcript[-1]

        if last_character not in ['.', '!', '?', ']']:
            clean_transcript = False 
        else :
            clean_transcript = True
    
    if clean_transcript: 
        print("Test : Transcripts are correct - PASS")
    else : 
        print ("Test : Transcripts are not correct - FAIL")