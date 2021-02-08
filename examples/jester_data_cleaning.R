##############################
### Jester Cleaning Script ###
##############################
# Jester is available here: http://eigentaste.berkeley.edu/dataset/

library(reshape2)
library(readxl)
library(tidyverse)

jester_data_1 <- read_excel("path_to/jester-data-1.xls", col_names = FALSE)
jester_data_2 <- read_excel("path_to/jester-data-2.xls", col_names = FALSE)
jester_data_3 <- read_excel("path_to/jester-data-3.xls", col_names = FALSE)

# Stack those into one matrix
jester = rbind(jester_data_1, jester_data_2, jester_data_3)

# Remove the first column (it's a count of ratings/user)
jester = jester[,-1]
jester[1:5, 90:101] # a little visual inspection
dim(jester) # chacking dimensions
colnames(jester) = paste0("i",seq(1,ncol(jester),1)) # just making sure everything lines up

jester$user = seq(1,nrow(jester),1) # making a "user" columns

j_long = melt(jester, id.vars=c("user")) # changing to long formart: we don't want a matrix, 
# we want (user, item, ratings) so we can use the built-in data input in Lenskit 

j_long = j_long %>% mutate(rating = value) %>% filter(rating != 99) # removing ratings of 99 (i.e. "did not rate")
j_long = j_long %>% select(user, item = variable, rating) # renaming variable column to item

length(unique(j_long$user)) # checking to make sure all the numbers add up to what is stated on the webpage
length(unique(j_long$item))
nrow(j_long)

# indexing the items
item_indexing = data.frame(new_index = 1:length(unique(j_long$item)), item = unique(j_long$item))

# putting the indices on
j_long = inner_join(j_long, item_indexing, by = c("item" = "item")) %>% select(user, item = new_index, rating)

# write the csv 
write_csv(path = "path/jester.csv", x = j_long)

######################
### Cleaning below ###
######################
# Assuming we have Jester in (user, item, rating) format
# We want to scale ratings from [-10, 10] onto the [1,5] scale
# (this makes things work better with CIBCF and CUBCF)

library(readr)
ratings <- read_csv("path_to/ratings.csv")

a = -10
b = 10
c = 1
d = 5

# do the transformation
ratings = ratings %>% 
            rowwise() %>% 
            mutate(rating = (rating - a)*(d - c)/(b - a) + c)


# write the csv (you don't have to overwrite the old one)
write_csv(path = "path_to/ratings.csv", x = ratings)


