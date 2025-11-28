library(tidySEM)
library(lavaan)
library(semPlot)
library(readxl)

data <- read_excel("data_path")
data_std <- as.data.frame(scale(data))

model <-'
        Swap ~  FS 
        Swap ~  GQ + P +  GA + LI
        Swap ~  Pre + Spei + Fvc + Temp 
        
        FS =~ GT + ALT
        ALT ~ Pre + Spei + Fvc + Temp 
        GT  ~ Pre + Spei + Fvc + Temp 
        
        ALT ~ Y
        GT ~ Y

        GQ ~ Y
        LI ~ Y
        GA ~ Y

        Pre ~ Y
        Spei ~ Y
        Fvc ~ Y
        Temp ~ Y
        GQ ~~ GA
'

fit <- sem(model, data_std, std.lv = TRUE)

summary(fit, fit.measures=TRUE, standard=TRUE)

fitMeasures(fit,c("chisq","df","pvalue","gfi","cfi","rmr","srmr","rmsea"))

lay2 <- get_layout(NA, 'GA', NA, 'P', NA,
                   'GQ', NA, NA, NA, 'LI',
                   NA, NA, 'Swap', NA, NA,
                   NA, NA, NA, NA, NA,
                   NA, NA, 'FS', NA, NA,
                   NA, NA, NA, NA, NA,
                   NA, 'GT', NA, 'ALT', NA,
                   'Pre', NA, NA, NA, 'Temp',
                   NA, 'Spei', NA, 'Fvc', NA, rows = 9)

nodes <- get_nodes(fit)
edges <- get_edges(fit)

nodes$colour <- 'black'
nodes$label_colour <- 'black'
nodes$label_size <- 5
nodes$fill <- "white"
nodes$label_fill <- 'transparent'

nodes$rect_height[nodes$name == "SW"] <- 1.5
nodes$rect_width[nodes$name == "SW"] <- 2.0

edges$colour <- 'black'
edges$colour[edges$est < 0] <- 'red'
edges$colour[edges$est >= 0] <- 'blue'
edges$show[edges$from == edges$to] <- FALSE
edges$connect_from[edges$from == 'FS' & edges$to == 'GT'] <- 'bottom'
edges$connect_to[edges$from == 'FS' & edges$to == 'GT'] <- 'right'
edges$connect_to[edges$from == 'FS' & edges$to == 'ALT'] <- 'left'

edges$size <- abs(as.numeric(as.character(edges$est))) *1 +1

graph_sem(fit, 
          layout = lay2, 
          nodes = nodes, 
          edges = edges,
          rect_width = 1.2,
          rect_height = 1.2,
          ellipses_width = 1.2,
          ellipses_height = 1.2)