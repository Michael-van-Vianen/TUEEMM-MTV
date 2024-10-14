rm(list = ls()) # 
library(igraph) # 
set.seed(123)  # 
random_network <- watts.strogatz.game(1, 50, 1, .35, loops = FALSE, multiple = FALSE)
plot(random_network, main="Random network", edge.arrow.size=.4, vertex.label.color="black", vertex.label.dist=0, vertex.size=14,
     vertex.label.cex=.9)

set.seed(123)  # set seed
g <- watts.strogatz.game(1, 100, 4, 0.05)
average.path.length(g)
plot(g, main="Random thingy", edge.arrow.size=.4, vertex.label.color="black", vertex.label.dist=0, vertex.size=14,
     vertex.label.cex=.9)
transitivity(g, type="average")

# Calculate shortest paths from node 89
distances <- shortest.paths(g, v = 30)


vertex_colors <- ifelse(distances <= 3, "green", "black")

plot(g, main="Nodes within distance 3 from node 89",
     edge.arrow.size=.4, vertex.label.color="black", vertex.label.dist=0, vertex.size=14,
     vertex.label.cex=.9, vertex.color=vertex_colors)

avg_path_length <- average.path.length(g)
cat("Average path length:", avg_path_length, "\n")

avg_transitivity <- transitivity(g, type="average")
cat("Average transitivity:", avg_transitivity, "\n")


#################
# distance measure with weighted graphs

library(igraph)

# Create the Watts-Strogatz graph
set.seed(123)  # For reproducibility
g <- watts.strogatz.game(1, 100, 5, 0.05)

degrees <- degree(g)

# Assign edge weights based on degrees
E(g)$weight <- apply(get.edges(g, E(g)), 1, function(e) {
  degrees[e[1]] + degrees[e[2]]
})

weighted_distances <- shortest.paths(g, v = 30, weights = E(g)$weight)

threshold <- quantile(weighted_distances, probs = 0.3)

vertex_colors <- ifelse(weighted_distances <= threshold, "green", "black")

plot(g, main="Weighted Distance",
     edge.arrow.size = 0.4, vertex.label.color = "black", vertex.label.dist = 0,
     vertex.size = 7, vertex.label.cex = 0.7, vertex.color = vertex_colors,
     edge.width = E(g)$weight / max(E(g)$weight) * 2) 

selected_nodes <- which(weighted_distances <= threshold)
cat("Nodes within the weighted distance threshold:", selected_nodes, "\n")


#######
# distance measure resistance distance

# Compute the Laplacian matrix and convert it to a standard numeric matrix

set.seed(123)  # For reproducibility
g <- watts.strogatz.game(1, 100, 4, 0.05)

laplacian_matrix <- as.matrix(graph.laplacian(g, normalized = TRUE))

laplacian_pseudo_inv <- MASS::ginv(laplacian_matrix)

# Test from node 30 
node_of_interest <- 30
resistance_distances <- diag(laplacian_pseudo_inv) + laplacian_pseudo_inv[node_of_interest, node_of_interest] - 2 * laplacian_pseudo_inv[, node_of_interest]

# Define a threshold 
threshold <- quantile(resistance_distances, probs = 0.3)

vertex_colors <- ifelse(resistance_distances <= threshold, "green", "black")

plot(g, main = "Resistance Distance",
     edge.arrow.size = 0.4, vertex.label.color = "black", vertex.label.dist = 0,
     vertex.size = 7, vertex.label.cex = 0.7, vertex.color = vertex_colors)

selected_nodes <- which(resistance_distances <= threshold)
cat("Nodes within the resistance distance threshold:", selected_nodes, "\n")

















