library(xgboost)
library(Matrix)
library(caret)
library(plotly)
library(readxl)
library(gt)
library(pracma)
library(reshape2)
library(ggstatsplot)
library(tidyverse)

#==== Data Wrangling ====


set.seed(1)

#CNV_ch_df_sub <- readRDS('CNV_ch_df_sub.rds')


#CNV_ch_df_sub <- CNV_ch_df_sub |> 
#  filter(time > 0) |>
#  group_by(Condition, epoch, Participant) |>
#  mutate(unique_ID = cur_group_id()) |> 
#  ungroup()


#exclude_culs <- c(
#  'time', 'condition', 'Stim', 'unique_ID')

#group_n <- 2/32

#CNV_ch_df_sub <- CNV_ch_df_sub |> 
#  mutate(bin = ceiling(time / group_n)) |> 
#  group_by(bin, Condition, epoch, Participant) |>
#  mutate(across(-all_of(exclude_culs), mean)) |> 
#  ungroup()

#df <- CNV_ch_df_sub |> 
#  select(-time, -Stim, -condition) |> 
#  distinct()

#saveRDS(df, 'CNV_binned.rds')

df <- readRDS('CNV_binned.rds')

unique_IDs <- unique(df$unique_ID)

Train_IDs <- sample(unique_IDs, length(unique_IDs) * 0.75)


#---- Train Data ----
Train_df <- df |> 
  filter(unique_ID %in% Train_IDs)

Test_df <- df |> 
  filter(!unique_ID %in% Train_IDs) 

exclude_cols <- c('epoch', 'Participant', 'Condition', 'unique_ID', 'bin')

Train_df_long <- Train_df |> 
  pivot_longer(
    cols = -all_of(exclude_cols),
    names_to = 'Variable', 
    values_to = 'Value')

Train_df_wide <- Train_df_long |> 
  select(-unique_ID) |> 
  pivot_wider(
    names_from = c(Variable, bin), 
    values_from = Value) |> 
  select(-epoch, -Participant)

Test_df_long <- Test_df |>
  pivot_longer(
    cols = -all_of(exclude_cols),
    names_to = 'Variable', 
    values_to = 'Value')

Test_df_wide <- Test_df_long |>
  select(-unique_ID) |> 
  pivot_wider(
    names_from = c(Variable, bin), 
    values_from = Value) |> 
  select(-epoch, -Participant)


#==== XGBoost Model ====  
Train_label <- Train_df_wide |> 
  mutate(Condition = case_when(
    Condition == 'One' ~ 0,
    Condition == 'Two' ~ 1
  )) |> 
  select(Condition) |>
  as.matrix()

Test_label <- Test_df_wide |> 
  mutate(Condition = case_when(
    Condition == 'One' ~ 0,
    Condition == 'Two' ~ 1
  )) |>
  select(Condition) |>
  as.matrix()

Train_features <- Train_df_wide |> 
  select(-Condition) |> 
  as.matrix()

Test_features <- Test_df_wide |>
  select(-Condition) |> 
  as.matrix()

Train_matrix <- xgb.DMatrix(data = Train_features, label = Train_label)
Test_matrix <- xgb.DMatrix(data = Test_features, label = Test_label)
                              
#==== Train Model ====
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.2,
  max_depth = 16,
  gamma = 5,
  set.seed = 1)

model <- xgb.train(
  params = params, 
  data = Train_matrix,
  watchlist = list(train = Train_matrix, test = Test_matrix),
  nrounds = 200)

#==== Predict ====
pred_probs <- predict(model, Test_matrix)
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

actual_labels <- getinfo(Test_matrix, "label")
pred_class <- factor(pred_class, levels = c(0, 1))

confusion_matrix <- confusionMatrix(pred_class, factor(actual_labels))
confusion_matrix$overall['Accuracy']

cm_table <- as.data.frame(confusion_matrix$table) |> 
  mutate(Prediction = case_when(
    Prediction == 0 ~ 'One',
    Prediction == 1 ~ 'Two'
  )) |> 
  mutate(Reference = case_when(
    Reference == 0 ~ 'One',
    Reference == 1 ~ 'Two'
  ))

cm_table <- cm_table |> 
  group_by(Reference) |> 
  mutate(Percent = Freq / sum(Freq) * 100) |> 
  ungroup()

# Plot with percentages
ggplot(data = cm_table, aes(x = Prediction, y = Reference, fill = Percent)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f%%", Percent)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "Confusion Matrix (Percentages)",
    x = "Predicted", y = "Actual") +
  theme_minimal()
