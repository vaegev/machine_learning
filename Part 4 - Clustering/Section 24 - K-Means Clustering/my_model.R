dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# elbow method to find number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = "b", main = paste('clusters of clients'), xlab = 'number of clusters', ylab = 'wcss')

# applying k-means to mall data
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# vis clusters
library(cluster)
clusplot(X, 
         kmeans$cluster, 
         lines = 0, 
         shade = TRUE, 
         color = TRUE, 
         labels = 2, 
         plotchar = FALSE,
         span = TRUE,
         main = paste('clusters of clients'),
         xlab = 'annual income',
         ylab = 'spending score')