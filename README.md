## Heuristic Machine Learning for Graph Data

### 1. data

I  synthesized a set of toy network data. 

Credit card transaction record (toy_fraud.csv):                  
"event_id" ->   transaction order             
"from_id"  ->   transaction sender ID      
"to_id" ->      transaction receiver ID  
"amt" ->        transaction amount 
"event_type" -> transaction type (pay/cashout/transfer/repay)    
"from_type" ->  sender type (acct/merchant)      
"to_type"  ->   receiver type (acct/merchant) 
"from_fraud" -> sender annoated as fraud before
"to_fraud" ->   receiver annoated as fraud before 
"txn_time" ->   transaction time 
"isFraud"  ->   transaction annotaed as fraud
 
Account attributes (toy_nodes.csv):
"node_id" -> sender or receiver's ID
"node_type" -> acct or merchant
"node_fraud" -> sender or receiver's fraud histoy
"overdue_last_1M" -> sender or reveiver's overdue counts in last one month
"overdue_last_3M" -> sender or reveiver's overdue counts in last three months
"risk_pref_invest" -> the preference of taking risk (scale = 1~10)

### 2. algos

I integrated three algos: 1) Personalised Page Rank 2) node2vec 3) DBSCAN

### 3. executation & objective

"heuristic_execute.ipynb" is for RISKY GROUPS DETECTION by first ranking (PPR) and then clustering (node2vec+DBSCAN)

"toy_fraud.R" is for network visualization, trying QAP and Exponential Randaom Graph Model
