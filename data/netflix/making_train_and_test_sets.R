library(readr)
netflix_data <- read_csv("NetflixPrizeData/netflix_data.csv")
probe_set <- read_csv("NetflixPrizeData/probe_set.csv")

head(probe_set)
head(netflix_data)

### Train set ###
# (full set minus probe set)
ts = anti_join(netflix_data, probe_set, by = c("user" = "user", "item" = "item"))

nrow(netflix_data) == nrow(ts) + nrow(qs)
write_csv("NetflixPrizeData/train_set.csv", x = ts)


### Test set ###
# (probe_set )
qs = right_join(netflix_data, probe_set, by = c("user" = "user", "item" = "item"))

nrow(probe_set) == nrow(qs)
write_csv("NetflixPrizeData/test_set.csv", x = qs)

### Data summary ###
length(unique(netflix_data$user)) # 480189
length(unique(netflix_data$item)) # 17770
