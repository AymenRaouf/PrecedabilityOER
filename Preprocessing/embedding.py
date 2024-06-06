import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from gensim.models import KeyedVectors
from gensim.models import FastText
from gensim.test.utils import common_texts
import numpy as np
import rdflib
import os

def symbols_filter(word):
    word = word.replace("%e2%80%93","-")
    word = word.replace("%27", "'")
    word = word.replace("%2f", "/")
    word = word.replace("%e2%80%94", "-")
    word = word.replace("%2e", ".")
    word = word.replace("%26", '&')
    return word


def load_model(path, name = 'wikipedia2vec'):
    # dimensions = ["100", "300", "500"]
    print(f"Loading {name} model ...")
    wikipedia2vec = KeyedVectors.load_word2vec_format(path, binary = False)
    print(f"Loaded {name} model ...")
    return wikipedia2vec


def wikipedia2vec_embedding(model, concept, d):
    try :
        return model['ENTITY/'+concept]
    except KeyError:
        return np.zeros((1,d))
        
        
def node_embeddings(model, g, d, method = 'wikipedia2vec'):
    embeddings_concepts = {}
    concepts = []
    missing_concepts = []
    if method == 'wikipedia2vec':
        for s, p, o in g:
            concept = symbols_filter(s.split('/')[-1])
            if str(p) == 'https://univ-nantes.fr/ontology/pageRank' :
                embedding = wikipedia2vec_embedding(model, concept, d)
                if len(embedding) != 0:
                    embeddings_concepts[concept] = {}
                    embeddings_concepts[concept]['list'] = embedding
                    embeddings_concepts[concept]['pageRank'] = float(o)
                    concepts.append(concept)
                else :
                    missing_concepts.append(concept)

    return {
        'embeddings' : embeddings_concepts,
        'concepts' : concepts,
        'missing_concepts' : missing_concepts
    }
    

def embedder_embeddings(resources, publisher, d = 300):
    path = "../Models/enwiki_20180420_"+str(d)+"d.txt"
    model = load_model(path, d)
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    concepts = []
    missing_concepts = []
    sentences_embedder = []
    for c in resources:
        embeddings_concepts = []
        current_concepts = []
        
        g = rdflib.Graph()
        g_path = os.path.join(script_dir, '../Output/Graphs/v01/' + publisher + '/' + str(c) + '.ttl')
        try :
            g.parse(g_path, format='turtle')

            embeddings = node_embeddings(model, g, d, method='wikipedia2vec')

            embeddings_concepts = embeddings['embeddings']
            missing_concepts.append(embeddings['missing_concepts'])
            current_concepts = embeddings['concepts']
            concepts.append(current_concepts)
            pageRankSum = np.sum([embeddings_concepts[k]['pageRank'] for k in embeddings_concepts], 0)
            embeddingsSum = np.sum([np.dot(embeddings_concepts[k]['list'], embeddings_concepts[k]['pageRank']) for k in embeddings_concepts], 0)
            sentences_embedder.append(embeddingsSum / pageRankSum)
        except:
            embeddingsSum = np.zeros((1,d))
            pageRankSum = np.ones((1,1))
            print(c, "not found")
            sentences_embedder.append((embeddingsSum / pageRankSum)[0])

    return sentences_embedder

def bert_embeddings(sentences):
    model_bert = RepresentationModel(
        model_type = "bert",
        model_name = "bert-base-uncased",
        use_cuda = False
    )
    sentences_vector_bert = list(model_bert.encode_sentences(sentences, combine_strategy = "mean"))
    
    return sentences_vector_bert

def fasttext_embeddings(sentences, n_gram, window, size, epochs):
    fasttext = FastText(vector_size = size, window = window, min_count = 1, min_n = n_gram, sentences = common_texts, epochs = 10)
    sentences_vector_fasttext = list(fasttext.wv[sentences])
    return sentences_vector_fasttext

def embeddings(sentences, resources, methods, publisher, save = False, path = ''):

    embeddings_df = pd.DataFrame()

    if 'BERT' in methods:
        embeddings_df['BERT'] = bert_embeddings(sentences)
    if 'embedd-er' in methods:
        embeddings_df['EMBEDD-ER'] = embedder_embeddings(resources, publisher, d = 300)
    if 'FastText' in methods:
        embeddings_df['FastText'] = fasttext_embeddings(sentences, n_gram = 3, window = 5, size = 300, epochs = 10)

    if save:
        embeddings_df.to_csv(path+"embeddings.csv", sep = '|')

    return embeddings_df
