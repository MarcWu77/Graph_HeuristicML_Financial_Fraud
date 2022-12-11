## Heuristic Machine Learning for Graph Data
### 1. data
I  synthesized a set of toy network data. 
One is a credit card transaction record (toy_fraud.csv): 
     Schema      Meaning                      
 0   event_id    transaction order             
 1   from_id     transaction sender      
 2   to_id       transaction receiver   
 3   amt         transaction amount -> float
 4   event_type  transaction type (pay/cashout/transfer/repay)    
 5   from_type   sender type (acct/merchant)      
 6   to_type     receiver type (acct/merchant) 
 7   from_fraud  sender annoated as fraud before  -> binary
 8   to_fraud    receiver annoated as fraud before -> bianry
 9   txn_time    transaction time -> datetime64[ns]
 10  isFraud     transaction annotaed as fraud -> binary    
 
 and the other is account attributes (toy_nodes.csv).
