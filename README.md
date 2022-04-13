# spreading-activation
This is a content-based recommender system based on IMDb movie metadata tuned on the movielens dataset.
It uses a graph representation of the IMDb metadata and performs spreading activation using SPARQL queries (not the fastest solution admittedly)

There are three steps in the process, all resolved within this repo
  1. Creation of the graph representation
  2. Spreading activation
  3. Hyperparameter tuning

## Graph representation
Happens in `preprocess.py`.

The graph is created using [Cinemagoer](https://cinemagoer.github.io/) and converted into [RDF turtle](https://www.w3.org/TR/turtle/)
using [rdflib](https://rdflib.readthedocs.io/en/stable/). The movielens movies are first searched for using cinemagoer to get their
IMDb movie ID and that ID is then queried through cinemagoer to get the complete movie object, from which the selected metadata is
created into the final graph. The algorithm creates graph nodes for each of the following:
  - Movie
  - Person (Actor, Director, Writer, Editor, Composer, Producer)
  - Genre
  - Language
  - Country of origin
  - Decade
 
 To run this step, place your movielens file (or movies you wantto use in the same format) in the file specified by the `ML_LOCATION` and `ML_VERSION` constants
 (when making this I didn't plan to do it more than once :) ). Run it using `python preprocess.py` Next, go outside for three hours, it takes a while.
 This is mostly due to the Cinemagoer double movie querying, because the initial search does not provide the full information, so it has to be done twice. 
 
 I would recommend you skip this step if possible and use the graph in `movielens/ml-1m/imdb-1m_2.ttl`.
 
 ## Spreading activation
 Happens in `spreading_activation.py`.
 
 Uses the `Spreader` class, which parses the graph created in step 1 and performs SA on it. To use the spreader:
  1. (edit the ttl file location if not using default)
  2. create new `Spreader` instance
  3. Set config if not using default using `Spreader.update_cfg(config)` (this is mostly for tuning)
  4. Prepare ids of movies to activate into list (using movielens id)
  5. Perform spreading using `Spreader.spread(movies_to_activate)`
  6. Get top_k results using `Spreader.get_top_k_as_list(k)`
    - Alternatively log using `Spreader.log_results(k)`

The initial parsing takes around 15-20 seconds due to the size of the graph, the subsequent activation roughly the same. If you need to perform multiple
SA, use thesame spreader object, the graph will reset its previous activations much faster than parsing again.
 
### The spreader config
The config dictionary is used to pass hyperparameters for the spreading activation. The values to set are the following
  - `activation_threshold`: How much activation a node needs to spread
  - `decay_factor`: What percentage of activation arrives at the destination node
  - `edge_weights`: Dictionary of edges across which activation is spread, and their decay

Looking at it with hindsight, `decay_factor` is probably redundant, as it could be replaced by lowering all edge weights, but I'm not changing it now.

## Hyperparamter tuning
Happens in `training.py` and `runs_vis.ipynb`.

The hyperparameters were tuned using [Hyperopt](https://hyperopt.github.io/hyperopt/), logged using [MLflow](https://mlflow.org/) 
and visualised using [seaborn](https://seaborn.pydata.org/).

Training used hyperopt's random search using uniform distribution on each value of the config (`cfg_space`). Scoring was performed using NDCG.
Runs are logged using MLflow for each config. To get `runs.csv` for your new runs, go to MLflow UI and download csv there (or another way but this is how I did it).
To get more than the default 100 you have to select more runs.

The hyperparams were trained on only a few UIDs. This is not ideal and ideally they should be trained on a significantly larger set of UIDs,
probably being rotated. Due to time constraints I did not do that. Validation data only used one fold, again, ideally use proper cross validation.

During training, I activated movies that recieved a rating of 3 or more (line 27). This could also be considered a hyperparameter, but I didn't tune it.
Maybe consider using 4 or 5 as a threshold?

Hyperopt's [`fmin`](https://github.com/hyperopt/hyperopt/wiki/FMin) function is what performs the tuning. I pass it the `spread_and_rate` function, which 
oversees logging, spreading and rating for the selected UIDs. For each UID, `rate_for_uid` is what performs the individual recommendation and rating.








 
