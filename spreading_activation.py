from os import path

from rdflib import Graph, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery

ML_VERSION = '1m'
ML_LOCATION = f'./movielens/ml-{ML_VERSION}/movies.dat'
TTL_FILE = f'./movielens/ml-{ML_VERSION}/imdb-{ML_VERSION}_2.ttl'
MV_URI_PREFIX = 'https://www.imdb.com/title/tt'

INIT_ACTIVATION = 1  # Activation given to initial set of nodes

HAS_ACT_QUERY = prepareQuery('SELECT ?act WHERE {?node <https://example.org/hasActivation> ?act}')
TRIPLES_QUERY = prepareQuery('SELECT ?p ?target WHERE {{ ?node ?p ?target } UNION {?target ?p ?node}}')
DEFAULT_CFG = {
        'activation_threshold': 0.31649221592381765,
        'decay_factor': 0.1068994469731867,
        'edge_weights': {  # Proportional amount of activation that will flow along the edge
            'https://example.org/fromDecade': 0.21802592960018782,
            'https://schema.org/Actor': 0.7099285518684758,
            'https://schema.org/Director': 0.3683726597328457,
            'https://schema.org/Writer': 0.5424443369826616,
            'https://schema.org/Producer': 0.7980341978743082,
            'https://schema.org/Editor': 0.42905888133528514,
            'https://schema.org/Genre': 0.13658013467165064,
            'https://schema.org/CountryOfOrigin': 0.33552755021787994,
            'https://schema.org/inLanguage': 0.6412984290406345,
        }
    }


def parse_graph(ttl_file=TTL_FILE):
    """
    Reads a TTL graph from the disc
    :return: rfdlib parsed graph
    """
    if not path.isfile(ttl_file):
        print(f'"{ttl_file}" not found')
        exit()
    print(f'ttl file found at "{ttl_file}"')
    g = Graph()
    g.parse(ttl_file)
    print('Graph parsed successfully')
    return g


class Spreader:
    def __init__(self, graph=None):
        if not graph:
            self.graph = parse_graph()
        else:
            self.graph = graph
        self.edge_weights = DEFAULT_CFG['edge_weights']
        self.decay_factor = DEFAULT_CFG['decay_factor']
        self.activation_threshold = DEFAULT_CFG['activation_threshold']
        self.initial_uris = None

    def update_cfg(self, cfg):
        """
        Set/Update the hyperparameters of the spreader.
        :param cfg: new configuration dict
        """
        self.edge_weights = cfg['edge_weights']
        self.decay_factor = cfg['decay_factor']
        self.activation_threshold = cfg['activation_threshold']

    def spread(self, movies_to_activate, spread_steps=2):
        """
        Resets graph activation and performs spreading activation
        :param spread_steps: how many steps of spreading activation to perform
        :param movies_to_activate: list of movielens OIDs of movies to activate initially
        """
        if not self.edge_weights:
            print("spreader used before cfg was provided")
            return None
        uris_to_spread = self.ml_initial_activation(movies_to_activate, reset=True)
        already_spread = set()
        self.initial_uris = set()
        for uri in uris_to_spread:
            self.initial_uris.add(uri)
            already_spread.add(uri)
        for _ in range(spread_steps):
            uris_to_spread = self.spread_step(uris_to_spread, already_spread)

    def spread_step(self, nodes_to_spread, already_spread):
        """
        Perform on step of spreading activation. Will spread activation of all given nodes.
        Any node that reaches activation threshold for the first time will be added to output list
        :param nodes_to_spread: list of URIRefs to perform spread on
        :param already_spread: list of URIRefs of nodes that have already been activated
        :return: nodes that have been activated above the threshold for the first time
        """
        spread_next = []
        for node_uri in nodes_to_spread:
            # Query finds all nodes that have outgoing or incoming edges to current node
            triples = self.graph.query(TRIPLES_QUERY, initBindings={'node': node_uri})
            nodes_to_activate = []  # List of nodes that will be sent activation
            current_activation = 0  # This will be found and set

            for pred, obj in triples:  # Find which nodes to send to and find current activation
                if str(pred) == 'https://example.org/hasActivation':
                    current_activation = float(obj)  # Unpack Literal
                elif pred not in already_spread and pred not in spread_next and (str(pred) in self.edge_weights):
                    nodes_to_activate.append((pred, obj))
            already_spread.add(node_uri)  # So as to not spread twice

            for pred, obj in nodes_to_activate:  # Send activation to found nodes and mark new ones for spread
                act_to_send = self.edge_weights[str(pred)] * current_activation
                node_act = 0  # Must iterate over Result, but there is only one item
                for line in self.graph.query(HAS_ACT_QUERY, initBindings={'node': obj}):
                    if line:
                        node_act = float(line[0])  # doesn't exist if node wasn't activated already
                new_activation = min(node_act + (act_to_send * self.decay_factor), 1)  # Max activation is 1
                if (new_activation > self.activation_threshold) and (obj not in already_spread):
                    spread_next.append(obj)
                    already_spread.add(obj)
                self.set_activation(obj, new_activation)
        return spread_next

    def log_results(self, top_k=20):
        top_k = self.get_top_k(top_k)
        for movie, ml_oid, activation in top_k:
            print(f'[{str(ml_oid)}]:\t{str(movie)}: {activation}')

    def get_top_k(self, k):
        """
        Returns list of most activated items that were not in the initial activation set
        :param k: How many recommended items to get
        :return: list of recommended items (movie uri, ml OID, activation)
        """
        return self.graph.query(  # Query not prepared as this is not called often
            f'''
            SELECT ?mov ?mloid ?act
            WHERE {{
                ?mov a schema:Movie .
                ?mov <https://example.org/ml-OID> ?mloid .
                ?mov <https://example.org/hasActivation> ?act .
                FILTER ( ?mov NOT IN ({', '.join(['<' + str(uri) + '>' for uri in self.initial_uris])}) )
            }}
            ORDER BY DESC(?act)
            LIMIT {k}
            '''
        )

    def get_top_k_as_list(self, k):
        return [(rec[0], rec[1], rec[2]) for rec in self.get_top_k(k)]

    def initial_activation(self, uris_to_activate, reset=False):
        """
        Set initial activation to INIT_ACTIVATION for selected movies, rest are unset
        :param reset: if True, all non-initial nodes will be set to 0 (much slower)
        :param uris_to_activate: list of IMDB movie IDS to activate
        :return: graph with activations
        """
        if True or reset:  # Set activation of all nodes to 0
            self.graph.update(f'''
            DELETE {{ ?s <https://example.org/hasActivation> ?act }}
            INSERT {{ ?s <https://example.org/hasActivation> {Literal(0)} }}
            WHERE {{ ?s a ?p }}            
            ''')
        for uri in uris_to_activate:
            self.set_activation(uri, INIT_ACTIVATION)  # this is faster than a SPARQL query
        return uris_to_activate

    def ml_initial_activation(self, oids_to_activate, reset=False):
        """
        Set initial activation to INIT_ACTIVATION for selected movies, rest are unset.
        First finds uris of provided OIDs and then calls initial_activation()
        :param reset: if True, all non-initial nodes will be set to 0 (much slower)
        :param oids_to_activate: list of movielens movie IDs to activate
        :return: graph with activations
        """
        uris_to_activate = set()
        for oid in oids_to_activate:
            query = f'SELECT ?s WHERE {{ ?s <https://example.org/ml-OID> {oid} }}'
            uri = None
            for u in self.graph.query(query):
                uri = u[0]
            if not uri:
                continue
            uris_to_activate.add(uri)  # There should only be one
        return self.initial_activation(uris_to_activate, reset=reset)

    def set_activation(self, uri, act):
        self.graph.set((
            URIRef(uri),
            URIRef('https://example.org/hasActivation'),
            Literal(act)
        ))
