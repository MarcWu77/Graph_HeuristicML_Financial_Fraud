## Heuristic Machine Learning for Graph Data

### 1. data & objective

I synthesized a set of toy fraud transactions data. The objective is to help risk control sector of a bank detecting fraud transactions OR account. Given transactions are naturally dyads and relational data, we may need graph related machine learning method to build the models.

Data schema explanation is stored in data/schema.txt.

### 2. algos

I integrated three algos: 1) Personalised Page Rank 2) node2vec 3) DBSCAN

### 3. executation

"heuristic_execute.ipynb" is for RISKY GROUPS DETECTION by first ranking (PPR) and then clustering (node2vec+DBSCAN)

"toy_fraud.R" is for network visualization, trying QAP and Exponential Randaom Graph Model

These two files can be run directly in your studio.
