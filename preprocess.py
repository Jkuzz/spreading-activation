import os.path
import sys
from urllib.parse import quote_plus

from imdb import Cinemagoer
from rdflib import Graph, URIRef, Literal, RDF
from timeit import default_timer as timer

ML_VERSION = '1m'
ML_LOCATION = f'./movielens/ml-{ML_VERSION}/movies.dat'
TTL_FILE = f'./movielens/ml-{ML_VERSION}/imdb-{ML_VERSION}_2.ttl'


def read_movielens(path, limit=sys.maxsize):
    """Reads the movielens dataset
    :param path: to the saved dataset
    :param limit: limit number of movies fetched (mostly for debug)
    :returns list of tuples (ml_id, ml_title)"""
    found_movies = []
    with open(path, 'r') as f:
        i = 0
        for line in f:
            movie_id, movie_title, _ = line.strip().split('::')
            found_movies.append((movie_id, movie_title))
            i += 1
            if i >= limit:
                return found_movies
    return found_movies


def fetch_movies(movies, log=True):
    """Searches for movie using Cinemagoer, saves found movie into list"""
    ia = Cinemagoer()
    fetched_movies = {}

    logging_frequency = len(movies) / 50  # For logging progress
    last_logged = 0
    time_started = timer()

    for i, movie in enumerate(movies):
        searched_movie = ia.search_movie(movie[1])  # search_movie returns an incomplete dict, this is to get id
        if len(searched_movie) == 0:  # Cinemagoer didn't find a match
            name_without_pars = movie[1][:movie[1].find('(')]  # Often does find if parentheses are removed
            print(f'\tCould find movie \"{movie[1]}\" in IMDB, looking for \"{name_without_pars}\"')
            searched_movie = ia.search_movie(name_without_pars)  # trying again
            if len(searched_movie) == 0:
                print(f'\tCould not find movie \"{name_without_pars}\" in IMDB, skipped')
                continue
            print(f"\tAdded \"{name_without_pars}\"")

        try:
            fetched_movie = ia.get_movie(searched_movie[0].movieID)  # must use get_movie to get complete info
        except Exception as e:
            print("Exception during fetching movie, skipping")
            print(e)
            continue
        fetched_movies[int(movie[0])] = fetched_movie

        if log and i >= last_logged + logging_frequency:  # Log fetching progress
            last_logged += logging_frequency
            time_taken = round((timer() - time_started) / 60)
            print(f"Fetching movies {round(100 * last_logged / len(movies))}% done, time taken {time_taken}min.")
    return fetched_movies


def rdf_serialise(movies):
    """
    Construct an RDF Turtle representation of input movies using their IMDB metadata
    :param movies: list of cinemagoer movie objects
    :return: Graph serialisation
    """
    g = Graph()
    concepts_to_add = [
        {'key': 'director', 'predicate': 'https://schema.org/Director', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'composer', 'predicate': 'https://schema.org/Composer', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'writer', 'predicate': 'https://schema.org/Writer', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'producer', 'predicate': 'https://schema.org/Producer', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'editor', 'predicate': 'https://schema.org/Editor', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'cast', 'predicate': 'https://schema.org/Actor', 'uri_prefix': 'https://www.imdb.com/name/nm',
         'obj_type': 'https://schema.org/Person'},
        {'key': 'genres', 'predicate': 'https://schema.org/Genre', 'uri_prefix': 'https://www.imdb.com/genre/',
         'obj_type': 'https://schema.org/Genre'},
        {'key': 'languages', 'predicate': 'https://schema.org/inLanguage', 'uri_prefix': 'https://www.imdb.com/langs/',
         'obj_type': 'https://schema.org/Language'},
        {'key': 'countries', 'predicate': 'https://schema.org/countryOfOrigin',
         'uri_prefix': 'https://www.imdb.com/countries/', 'obj_type': 'https://schema.org/Country'},
    ]
    for ml_oid, movie in movies.items():
        try:
            if not movie['title']:
                print('\tMovie missing title, skipping')
                continue
            if not movie['year']:
                print(f'\tMissing year for movie {movie["title"]}')
                continue
            current_uri = f'https://www.imdb.com/title/tt{movie.movieID}'
            basic_movie_info = [
                {'predicate': RDF.type, 'object': URIRef('https://schema.org/Movie')},
                {'predicate': URIRef('https://schema.org/Name'), 'object': Literal(movie['title'])},
                {'predicate': URIRef('https://schema.org/datePublished'), 'object': Literal(movie['year'])},
                {'predicate': URIRef('https://example.org/ml-OID'), 'object': Literal(ml_oid)},
            ]
            for info in basic_movie_info:
                g.add((  # Create basic movie predicates
                    URIRef(current_uri),
                    info['predicate'],
                    info['object']
                ))
            for concept in concepts_to_add:  # Create graph edges
                if concept['key'] in movie.keys():
                    for item in movie[concept['key']]:
                        if not item:
                            continue  # Sometimes there was a None
                        uri = concept['uri_prefix'] + quote_plus(try_get_id(item))
                        rdf_insert_named(g, concept['obj_type'], try_get_name(item), uri)
                        g.add((
                            URIRef(current_uri),
                            URIRef(concept['predicate']),
                            URIRef(uri)
                        ))
            if movie['year']:  # Decade aggregation edge
                g.add((
                    URIRef(current_uri),
                    URIRef('https://example.org/fromDecade'),
                    URIRef(get_decade(g, movie['year']))
                ))
        except Exception as e:
            print(f"Exception {e} occurred while serialising {movie['title']}, skipping movie")
            continue
    return g


def get_decade(graph, year):
    decade = year - year % 10
    decade_uri = f'https://www.imdb.com/decades/'
    return rdf_insert_named(graph, 'https://example.org/Decade', decade, f'{decade_uri}{decade}')


def try_get_id(obj):
    if not obj:
        return '0'
    if hasattr(obj, 'personID'):
        return obj.personID
    if hasattr(obj, 'movieID'):
        return obj.movieID
    if str(obj):
        return str(obj)
    return '0'


def try_get_name(obj):
    if hasattr(obj, 'name'):
        return obj.name
    return str(obj)


def rdf_insert_named(graph, obj_type, name, uri):
    graph.add((URIRef(uri), RDF.type, URIRef(obj_type)))
    graph.add((URIRef(uri), URIRef('https://schema.org/Name'), Literal(name)))
    return uri


def save_graph(graph, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(graph.serialize())


def main():
    if os.path.isfile(TTL_FILE):
        print(f'TTL file already exists at {TTL_FILE}, to run this script, change path or delete the file')
        exit()
    print(f'Reading movielens from "{ML_LOCATION}"...')
    ml_movies = read_movielens(ML_LOCATION)
    print(f'Read {len(ml_movies)} movies.')
    print('Fetching movie info using Cinemagoer...')
    found_movies = fetch_movies(ml_movies)
    print(f'IMDB info loaded, serialising to RDF...')
    graph = rdf_serialise(found_movies)
    print(f'RDF serialisation complete, saving to {TTL_FILE}...')
    save_graph(graph, TTL_FILE)


if __name__ == "__main__":
    main()
