library(tidyverse)
library(readr)
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)

# Pick the dataset that's being processed
dataset = 'ml1m' # jester, ml1m, ml10m, ml20m

if(dataset == "ml20m"){
  data_name = "ML20M"
}else if(dataset == "ml10m"){
  data_name = "ML10M"
}else if(dataset == "ml1m"){
  data_name = "ML1M"
}else if(dataset == "jester"){
  data_name = "Jester"
}

# For baselines
baseline_path = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/baselines/results")
setwd(baseline_path)
tbl_baselines <- list.files(pattern = "*.csv") %>% map_df(~read_csv(.))

if(dataset == "ml20m"){
  tbl_baselines = tbl_baselines %>% filter(name != 0)
}

baseline_names = data.frame(name = c("ItemKNN-Average", "UserKNN-Average"), 
                            Algorithm = c("IBCF","UBCF"), mysize = c(0.75))
tbl_baselines = inner_join(tbl_baselines, baseline_names, by = c("name" = "name")) %>% 
  filter(nnbrs != 10) %>% select(Algorithm, ndcg, precision, list_length, mysize, K = nnbrs)
tbl_baselines = tbl_baselines %>% filter(Algorithm != "UBCF")

tbl_baselines

max_list_length = 10

# For Confidence-aware Approaches
if(dataset == 'jester'){
  r0 = 2
}else{r0 = 4}  # 35 (3.5), 4, 45 (4.5), 5

path_cubcf = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/confidence_aware/CUBCF/r",r0)
setwd(path_cubcf)
#baseline_name = "UserKNN-Average"
tbl_conf_cubcf <- list.files(pattern = "*.csv") %>% map_df(~read_csv(.))

path_cibcf = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/confidence_aware/CIBCF/r",r0)
setwd(path_cibcf)
#baseline_name = "ItemKNN-Average"
tbl_conf_cibcf <- list.files(pattern = "*.csv") %>% map_df(~read_csv(.))


alg_names_cubcf = data.frame(name = c("UserKNN-CA-Average", "UserKNN-CA-BS-Average", "UserKNN-CA-JK-Average"), 
                             Algorithm = c("CUBCF", "CUBCF-BS", "CUBCF-JK"), mysize = c(0.75))

alg_names_cibcf = data.frame(name = c("ItemKNN-CA-Average", "ItemKNN-CA-BS-Average", "ItemKNN-CA-JK-Average"), 
                             Algorithm = c("CIBCF", "CIBCF-BS", "CIBCF-JK"), mysize = c(0.75))


tbl_conf_cubcf = inner_join(tbl_conf_cubcf, 
                            alg_names_cubcf, 
                            by = c("name" = "name")) %>% 
  filter(nnbrs != 10) %>% select(Algorithm, ndcg, precision, list_length, mysize, K = nnbrs)

tbl_conf_cibcf = inner_join(tbl_conf_cibcf, 
                            alg_names_cibcf, 
                            by = c("name" = "name")) %>% 
  filter(nnbrs != 10) %>% select(Algorithm, ndcg, precision, list_length, mysize, K = nnbrs)


# Precision
p3_20m = bind_rows(tbl_baselines, tbl_conf_cibcf, tbl_conf_cubcf) %>% 
  mutate(K = as.factor(K)) %>% 
  filter(list_length <= max_list_length) %>% filter(Algorithm %in% c("CIBCF-JK","CIBCF-BS","IBCF")) %>% 
  #filter(NumNbrs != 75, NumNbrs != 10, NumNbrs != 50) %>%
  ggplot(aes(x = list_length, y = precision, color = Algorithm, shape = K)) + #, shape = NumNbrs)) + 
  geom_line() + scale_size(range = c(0.5, 1.5), guide="none") + geom_point(size = 3.5) + 
  theme_bw() + ggtitle(paste("Precision -", data_name)) + 
  theme(text = element_text(size=20))

# NDCG
n3_20m = bind_rows(tbl_baselines, tbl_conf_cibcf, tbl_conf_cubcf) %>% 
  mutate(K = as.factor(K)) %>% 
  filter(list_length <= max_list_length) %>% filter(Algorithm %in% c("CUBCF-BS","CUBCF-JK","CUBCF","IBCF")) %>% 
  #filter(NumNbrs != 75, NumNbrs != 10, NumNbrs != 50) %>%
  ggplot(aes(x = list_length, y = ndcg, color = Algorithm, shape = K)) + #, shape = NumNbrs)) + 
  geom_line() + scale_size(range = c(0.5, 1.5), guide="none") + geom_point(size = 3.5) + 
  theme_bw() + ggtitle(paste("NDCG -", data_name)) + 
  theme(text = element_text(size=20))
  
  



#legend_p1 = gtable_filter(ggplot_gtable(ggplot_build(p1_1m)), "guide-box")
#legend_p2 = gtable_filter(ggplot_gtable(ggplot_build(p2_1m)), "guide-box")
legend_p3 = gtable_filter(ggplot_gtable(ggplot_build(p3_1m)), "guide-box")
#legend_n1 = gtable_filter(ggplot_gtable(ggplot_build(n1_1m)), "guide-box")
#legend_n2 = gtable_filter(ggplot_gtable(ggplot_build(n2_1m)), "guide-box")
legend_n3 = gtable_filter(ggplot_gtable(ggplot_build(n3_20m)), "guide-box")


library(ggpubr)

p1_n = n3_1m + xlab("List Length (k)") + 
  scale_x_continuous(breaks = seq(1,10)) + ylab("NDCG@k") + 
  theme(text = element_text(size=25), legend.title = element_text(size = 15))
p2_n = n3_10m + xlab("List Length (k)") + 
  scale_x_continuous(breaks = seq(1,10)) + ylab("NDCG@k") + 
  theme( text = element_text(size=25), legend.title = element_text(size = 15))
p3_n = n3_20m + xlab("List Length (k)") + 
  scale_x_continuous(breaks = seq(1,10)) + ylab("NDCG@k") + 
  theme(text = element_text(size=25), legend.title = element_text(size = 15))
p4_n = n3_j + xlab("List Length (k)") +
  scale_x_continuous(breaks = seq(1,10)) + ylab("NDCG@k") + 
  theme(text = element_text(size=25), legend.title = element_text(size = 15))

p1_p = p3_1m + xlab("List Length (k)") +  
  scale_x_continuous(breaks = seq(1,10)) + ylab("Precision@k") + 
  theme(text = element_text(size=25))
p2_p = p3_10m + xlab("List Length (k)") + ylab("Precision@k") + 
  scale_x_continuous(breaks = seq(1,10)) + 
  theme(text = element_text(size=25))
p3_p = p3_20m + xlab("List Length (k)") + ylab("Precision@k") + 
  scale_x_continuous(breaks = seq(1,10)) + 
  theme(text = element_text(size=25))
p4_p = p3_j + xlab("List Length (k)") + ylab("Precision@k") + 
  scale_x_continuous(breaks = seq(1,10)) + 
  theme(text = element_text(size=25))

ggarrange(p1_p, p2_p, p3_p, p4_p, 
          ncol = 1, nrow = 4, 
          common.legend = TRUE, legend="bottom")

ggarrange(p1_n, p2_n, p3_n, p4_n, 
          ncol = 1, nrow = 4, 
          common.legend = TRUE, legend="bottom")



