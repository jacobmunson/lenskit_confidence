library(readr)

dataset = 'jester' #'ml20m' #'jester' #"ml1m" # ml1m
# For baselines
baseline_path = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/baselines/results")
setwd(baseline_path)

for(list_length in c(1,2,3,4,5,6,7,8,9,10)){ #,20,30,40,50,60,70,80,90,100)){
  print(list_length)
  csv_path = paste0("results",list_length,".csv")
  
  results_df <- read_csv(csv_path)
  
  results_df$list_length = list_length
  results_df
  
  write_csv(results_df, csv_path)
}


### Confidence-aware
r0 = 2 # 35 (3.5), 4, 45 (4.5), 5
baseline = "IBCF" # UBCF or IBCF
if(baseline == "UBCF"){
  path = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/confidence_aware/CUBCF/r",r0)
  baseline_name = "UserKNN-Average"
}else if(baseline == "IBCF"){
  path = paste0("D:/results_conf_aware_nbhd_KDD2021/",dataset,"/confidence_aware/CIBCF/r",r0)
  baseline_name = "ItemKNN-Average"
}; setwd(path)


for(list_length in c(1,2,3,4,5,6,7,8,9,10)){ #},20,30,40,50,60,70,80,90,100)){
  print(list_length)
  csv_path = paste0("results",list_length,".csv")
  
  results_df <- read_csv(csv_path)
  
  results_df$list_length = list_length
  results_df
  
  write_csv(results_df, csv_path)
}

