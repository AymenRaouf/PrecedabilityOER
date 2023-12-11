import urllib.request
from rdflib import *
import urllib.parse
import json
import time


def segment_text(text, max_characters):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        if sum(len(w) for w in current_segment) + len(current_segment) + len(word) <= max_characters:
            current_segment.append(word)
        else:
            segments.append(" ".join(current_segment))
            current_segment = [word]

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments

def text2Wikifier(text, threshold, lang="en"):
    '''
    Calls wikifier on the text
    Input :
        text : input text to wikify
        threshold : pageRank threshold to use
        lang : language to use
    Output :
        Dictionary containing Wikifier's response
    '''

    results = []
    url = "http://www.wikifier.org/annotate-article"
    max_characters = 10000
    texts = segment_text(text, max_characters)
    
    for text in texts :
        # prepare the request
        data = urllib.parse.urlencode([
            ("text", text), ("lang", lang),
            ("userKey", "czgzhhslecgukacypvglbujzfmsdkm"),
            ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
            ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
            ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
            ("support", "false"), ("ranges", "false"), ("minLinkFrequency", "2"),
            ("includeCosines", "true"), ("maxMentionEntropy", "3")
            ])

        # call Wikifier and read the response.
        req = urllib.request.Request(url, data = data.encode("utf8"), method = "POST")
        with urllib.request.urlopen(req, timeout = 60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
            results = results + response["annotations"]

    # return the Wikifier response if it contains a body (concepts)
    try :
        return results
    except KeyError :
        return None


def wikifier2RDF(data):
    '''
    Creates RDF graph from Wikifier response 
    Input :
        data : Wikifier response 
    Output :
        RDF graph
    '''
    # if no data is provided then exit
    if data == None:
        return None

    # create graph and add namespaces
    g = Graph()
    EX = Namespace("http://example.org/")
    SC = Namespace("http://schema.org/")
    UN = Namespace("https://univ-nantes.fr/ontology/")
    DCT = Namespace("http://purl.org/dc/terms/")
    WIKI = Namespace("http://en.wikipedia.org/wiki/")
    DBPR = Namespace("https://dbpedia.org/resource/")
    DBPT = Namespace("https://dbpedia.org/ontology/")
    g.bind("", EX)
    g.bind("sc", SC)
    g.bind("dct", DCT)
    g.bind("wiki", WIKI)
    g.bind("dbpr", DBPR)
    g.bind("dbpt", DBPT)    

    # adding triples to the graph
    for concept in data:
        # we can add other metadata or info here if we want
        g.add((URIRef("http://example.org/Chapter"), URIRef(DCT.subject), URIRef(concept["url"])))
        g.add((URIRef(concept["url"]), URIRef(DCT.title), Literal(concept["title"])))
        g.add((URIRef(concept["url"]), URIRef(UN.pageRank), Literal(concept["pageRank"])))
        #g.add((URIRef(concept["url"]), URIRef(UN.cosine), Literal(concept["cosine"])))
        '''if len(concept["wikiDataClasses"]) > 0 :
            for label in concept["wikiDataClasses"]:
                g.add((URIRef(concept["url"]), URIRef(RDFS.subClassOf), Literal("http://en.wikipedia.org/wiki/"+label["enLabel"])))
        '''
        if len(concept["dbPediaTypes"]) > 0 :
            for label in concept["dbPediaTypes"]:
                g.add((URIRef(concept["url"]), URIRef(RDFS.subClassOf), URIRef("https://dbpedia.org/ontology/"+label)))
    return g


def text2RDF(text, path, threshold = -1):
    '''
    Creates RDF graph from text (pipeline)
    Input :
        text : input text to transform
        path : path in which the RDF graph will be saved
        threshold : pageRank threshold to use
    Output :
        RDF graph
    '''
    
    done = False
    # extract the name of the directory (graph)
    name = path.split("/")[-1].split(".")[0]
    while not done:
        try :
            # wikification
            print("Wikification for : " + name)
            data = text2Wikifier(text, threshold)
            
            # graph generation from wikifier results
            print("Graph generation for : " + name)
            g = wikifier2RDF(data)

            if g == None :
                print("There are no elements inside this chapter !")
                return None
            print("Saved in ", path + '.ttl')
            g.serialize(format = "turtle", destination = path + '.ttl')
            done = True
        except Exception as e:
            print('Runtime Error : ', e)
            print('Sleeping for 10 seconds...')
            time.sleep(10)
            print('Sleep done !')
    
    return g


