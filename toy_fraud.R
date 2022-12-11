#setwd("~/Desktop/PROJECT/working_zone/data/fraud")

library(statnet)
library(igraph)
library(sna)
library(ergm)

# 裝載數據
edge.list <- read.csv('data/toy_fraud.csv')
node.attr <- read.csv('data/toy_nodes.csv')
View(edge.list); View(node.attr)

#---DATA-PREPROCESS----
node.attr$node_type[node.attr$node_type == "acct"] <- 1
node.attr$node_type[node.attr$node_type == "merchant"] <- 2
node.attr$node_type <- as.integer(node.attr$node_type)

#----NETWORK-CONFIGURATION----
DG <- graph.data.frame(as.matrix(edge.list[,2:4]), node.attr, directed = TRUE)
vcount(DG)
ecount(DG)

#----SNA-VISUALIZATION----
grep('grey',colors(),value = T)

colors <- c('lightblue','palevioletred2')
#colors[V(DG)$node_type]
V(DG)$color <- colors[V(DG)$node_type]
V(DG)$size <- sqrt(degree(DG)) * 5
V(DG)$label <- NA

E(DG)$arrow.size <- 0.3
E(DG)$color <- 'darkgrey'
E(DG)$width <- log1p(edge.list$amt)/4

plot(DG, layout = layout.auto)

#----SNA-METRICS & UNDERSTANDING----
summary(degree(DG))
summary(closeness(DG, mode = 'total'))
summary(betweenness(DG))
closeness(DG, mode = 'total')

avg.local.metrics <- c(mean(degree(DG, mode = 'total')),
                       mean(degree(DG, mode = 'in')),
                       mean(degree(DG, mode = 'out')),
                       mean(closeness(DG, mode = 'total')),
                       mean(betweenness(DG,directed = TRUE, normalized = TRUE)),
                       mean(eigen_centrality(DG, directed = TRUE, scale = TRUE,
                                             weights = edge.list$amt)$vector))
avg.local.metrics
?centralization
global.centrality <- c(centralization.degree(DG, mode = 'total')$centralization,
                       centralization.degree(DG, mode = 'in')$centralization,
                       centralization.degree(DG, mode = 'out')$centralization,
                       centralization.closeness(DG, mode = 'total')$centralization,
                       centralization.betweenness(DG)$centralization,
                       centralization.evcent(DG)$centralization)
global.centrality

global.cohension <- c(graph.density(DG), 
                      transitivity(DG, type = 'average'), 
                      reciprocity(DG))
global.cohension

#----QAP----
E(DG)$weight <- log10(edge.list$amt)
transaction.matrix.1 <- as_adjacency_matrix(DG, 
                                          attr = 'weight', 
                                          sparse = FALSE) # Y1:交易紀錄金額加權矩陣
E(DG)$fraud <- edge.list$isFraud
transaction.matrix.2 <- as_adjacency_matrix(DG, 
                                          attr = 'fraud', 
                                          sparse = FALSE) # Y2:欺詐交易矩陣
#?dist
RiskPrefInvest.simi <- as.matrix(dist(node.attr$risk_pref_invest, 
                                      method = 'euclidean')) # X1:歐式距離
OverDueFreq.1M.simi <- as.matrix(dist(node.attr$overdue_last_1M,
                                      method = 'canberra')) # X2:坎培拉距離
OverDueFreq.3M.simi <- as.matrix(dist(node.attr$overdue_last_3M,
                                      method = 'canberra')) # X3:坎培拉距離
HistoryBad.prob <- as.matrix(dist(node.attr$node_fraud,
                                  method = 'binary')) # X4:binary距離

QAP.model.1 <- netlm(transaction.matrix.1, 
                   list(RiskPrefInvest.simi, 
                        OverDueFreq.1M.simi, 
                        OverDueFreq.3M.simi,
                        HistoryBad.prob),
                   mode = 'digraph',
                   test.statistic = 'beta',
                   nullhyp = 'qap',
                   intercept = TRUE,
                   reps = 500)
summary(QAP.model.1)

QAP.model.2 <- netlogit(transaction.matrix.2, 
                     list(RiskPrefInvest.simi, 
                          OverDueFreq.1M.simi, 
                          OverDueFreq.3M.simi,
                          HistoryBad.prob),
                     mode = 'digraph',
                     test.statistic = 'z-value',
                     nullhyp = 'qap',
                     intercept = TRUE,
                     reps = 500)
summary(QAP.model.2)

#----ERGM----
View(transaction.matrix.2)
## 配置網路
FraudNX <- network(transaction.matrix.2, 
                   directed = TRUE, loops = FALSE) # 構建交易欺詐網絡(Y)

## 配置節點屬性
for (attr.name in colnames(node.attr)[-1]) { # 在網絡中添加節點屬性(X)
  FraudNX %v% attr.name = node.attr[attr.name]
}

## 固定模型超參數
params = control.ergm(MCMC.samplesize = 3000, MCMC.burnin = 5000,
                      MCMLE.maxit = 2,# 模型最大迭代次數
                      seed = 2023) # 隨機種子數寫死

## 建模
#1. NULL MODLE
ERGM.NULL <- ergm(FraudNX ~ edges, control = params)
#2. BASELINE MODLE
#?ergm.terms
ERGM.nodeattr <- ergm(FraudNX ~ edges
                        # NODE NUMETRIC ATTRIBUTES (if categorical?)
                      + nodeocov('overdue_last_1M') 
                      + nodeocov('overdue_last_3M') # 交易發起方的逾期次數
                      + nodeicov('overdue_last_1M') 
                      + nodeicov('overdue_last_3M') # 交易接收方的逾期次數
                      + nodeocov('risk_pref_invest'), # 交易發起方的投資風險偏好
                      control = params) 
#2. EXPERIMENT MODLE 1
#?ergm.terms
ERGM.homophily <- ergm(FraudNX ~ edges
                         # NODE NUMETRIC ATTRIBUTES
                       + nodeocov('overdue_last_1M') + nodeocov('overdue_last_3M') 
                       + nodeicov('overdue_last_1M') + nodeicov('overdue_last_3M') + nodeocov('risk_pref_invest') 
                         # HOMOPHILY
                       + absdiff('risk_pref_invest',
                                 pow=0.5) # 節點投資風險偏好的歐式距離
                       + absdiff('node_type',pow=1) # 節點帳戶類型的相似度
                       + absdiff('node_fraud',pow=1), # 節點欺詐歷史的相似度
                       control = params) 
#3. EXPERIMENT MODLE 2
#?ergm.terms
ERGM.structure <- ergm(FraudNX ~ edges
                         # NODE NUMETRIC ATTRIBUTES
                       + nodeocov('overdue_last_1M') + nodeocov('overdue_last_3M') 
                       + nodeicov('overdue_last_1M') + nodeicov('overdue_last_3M') 
                       + nodeocov('risk_pref_invest') 
                         # HOMOPHILY
                       + absdiff('risk_pref_invest', pow=0.5) 
                       + absdiff('node_type',pow=1) 
                       + absdiff('node_fraud',pow=1)                       
                         # NETWORK STRUCTURE
                       + ostar(2) # 網路整體Engagement
                       + istar(2) # 網路整體Popularity
                       + mutual(diff=TRUE) # 網路整體Reciprocity
                       + triangle(attr=NULL) # 網路整體的三角關係數量(transitivity的近似指標)
                       + gwesp(0,fixed=TRUE), # 網絡整體的加權Transitivity
                       control = params) 

summary(ERGM.NULL) # AIC: 385.6  BIC: 391.7 Residual Dev 383.6
summary(ERGM.nodeattr) # AIC: 337.7  BIC: 374.7 Residual Dev:  325.7
summary(ERGM.homophily) # AIC: 311.5  BIC: 367.1 Residual Dev:  293.5
summary(ERGM.structure) # AIC: 283  BIC: 369.4 Residual Dev:  255

