import pandas as pd
from simpletransformers.language_representation import RepresentationModel


def bert_embeddings(sentences):
    model_bert = RepresentationModel(
        model_type = "bert",
        model_name = "bert-base-uncased",
        use_cuda = False
    )
    sentences_vector_bert = list(model_bert.encode_sentences(sentences, combine_strategy = "mean"))
    
    return sentences_vector_bert


def embeddings(sentences, methods, save = False, path = ''):

    embeddings_df = pd.DataFrame()

    if 'BERT' in methods:
        embeddings_df['BERT'] = bert_embeddings(sentences)

    if save:
        embeddings_df.to_csv(path+"embeddings.csv", sep = '|')

    return embeddings_df
