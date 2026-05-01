library(tidyverse)
library(readr)
library(gt)

# importing df
df_1 <- read_csv('../Participants/CNV_LSTM_3_1.csv')
df_2 <- read_csv('../Participants/CNV_LSTM_3_2.csv')

df <- bind_rows(df_1, df_2)



# creating accuracy cm 
total_One <- sum(df$total_One)
total_Two <- sum(df$total_Two)

correct_One <- sum(df$correct_One)
correct_Two <- sum(df$correct_Two)

true_One <- (correct_One/total_One) * 100
true_Two <- (correct_Two/total_Two) * 100
incorrect_One <- ((total_One - correct_One)/total_One) * 100
incorrect_Two <- ((total_Two - correct_Two)/total_Two) * 100


cm_df <- data.frame(
  Reference = c('One', 'One', 'Two', 'Two'),
  Prediction = c('One', 'Two', 'One', 'Two'),
  Percent = c(true_One, incorrect_One, incorrect_Two, true_Two)
)

cm_plot <- cm_df |> 
  mutate(Reference = case_when(
    Reference == 'One' ~ 'Straight',
    Reference == 'Two' ~ 'Diagonal'
  )) |> 
  mutate(Prediction = case_when(
    Prediction == 'One' ~ 'Straight',
    Prediction == 'Two' ~ 'Diagonal'
  )) |> 
  ggplot(aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Percent), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue",
                      limits = c(0,100)) +
  geom_text(aes(
    label = sprintf("%.1f%%", Percent)), color = "white", size = 6) +
  labs(title = "Confusion Matrix",
       x = "Actual Condition",
       y = "Predicted Condition",
       fill = "Percentage") +
  theme_minimal()
cm_plot

# creating accuracy violin plot
accuracy_df <- df |> 
  select(participant_id, accuracy_One, accuracy_Two) |> 
  mutate(
    accuracy_One = accuracy_One * 100,
    accuracy_Two = accuracy_Two * 100
  )

accuracy_viollin <- accuracy_df |> 
  pivot_longer(
    cols = c(accuracy_One, accuracy_Two),
    names_to = "Condition",
    values_to = "Accuracy"
  ) |> 
  mutate(Condition = case_when(
    Condition == 'accuracy_One' ~ 'Straight',
    Condition == 'accuracy_Two' ~ 'Diagonal'
  )) |> 
  ggplot(aes(x = Condition, y = Accuracy, fill = Condition)) +
  geom_violin(
    trim = FALSE, alpha = 0.5) +
  
  geom_boxplot(
    width = 0.1, position = position_dodge(0.9), outlier.shape = NA) +
  
  geom_jitter(aes(color = Condition),
              width = 0.1, size = 2, alpha = 0.7) +
  scale_color_manual(
    values = c("Straight" = "darkblue", "Diagonal" = "darkred")) +
  
  stat_summary(
    fun = mean, geom = "point", shape = 21, size = 5, fill = "white") +
  
  labs(title = "Participant Accuracy by Condition",
       x = "Condition",
       y = "Accuracy (%)") +
  theme_minimal() +
  theme(legend.position = "none")

accuracy_viollin

# sensitivity 

sensitivity_df <- df |> 
  mutate(
    sensitivity_One =
      correct_One / (correct_One + (total_Two - correct_Two)) * 100,
    sensitivity_Two =
      correct_Two / (correct_Two + (total_One - correct_One)) * 100) |> 
  select(participant_id, sensitivity_One, sensitivity_Two)


sensitivity_viollin <- sensitivity_df |>
  pivot_longer(
    cols = c(sensitivity_One, sensitivity_Two),
    names_to = "Condition",
    values_to = "Sensitivity"
  ) |> 
  mutate(Condition = case_when(
    Condition == 'sensitivity_One' ~ 'Straight',
    Condition == 'sensitivity_Two' ~ 'Diagonal'
  )) |> 
  ggplot(aes(x = Condition, y = Sensitivity, fill = Condition)) +
  geom_violin(
    trim = FALSE, alpha = 0.5) +
  
  geom_boxplot(
    width = 0.1, position = position_dodge(0.9), outlier.shape = NA) +
  
  geom_jitter(aes(color = Condition),
              width = 0.1, size = 2, alpha = 0.7) +
  scale_color_manual(
    values = c("Straight" = "darkblue", "Diagonal" = "darkred")) +
  
  stat_summary(
    fun = mean, geom = "point", shape = 21, size = 5, fill = "white") +
  
  labs(title = "Participant Sensitivity by Condition",
       x = "Condition",
       y = "Sensitivity (%)") +
  theme_minimal() +
  theme(legend.position = "none")
sensitivity_viollin

# specificity
specificity_df <- df |> 
  mutate(
    specificity_One =
      correct_Two / (correct_Two + (total_One - correct_One)) * 100,
    specificity_Two =
      correct_One / (correct_One + (total_Two - correct_Two)) * 100) |> 
  select(participant_id, specificity_One, specificity_Two)

specificity_viollin <- specificity_df |>
  pivot_longer(
    cols = c(specificity_One, specificity_Two),
    names_to = "Condition",
    values_to = "Specificity"
  ) |>
  mutate(Condition = case_when(
    Condition == 'specificity_One' ~ 'Straight',
    Condition == 'specificity_Two' ~ 'Diagonal'
  )) |>
  ggplot(aes(x = Condition, y = Specificity, fill = Condition)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(
    width = 0.1, position = position_dodge(0.9), outlier.shape = NA) +
  geom_jitter(aes(color = Condition),
              width = 0.1, size = 2, alpha = 0.7
  ) +
  scale_color_manual(
    values = c("Straight" = "darkblue", "Diagonal" = "darkred")
  ) +
  stat_summary(
    fun = mean, geom = "point", shape = 21, size = 5
    , fill = "white"
  ) +
  labs(title = "Participant Specificity by Condition",
       x = "Condition",
       y = "Specificity (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
specificity_viollin

# summary table 
summary_table <- data.frame(
  Metric = c('Accuracy', 'Sensitivity', 'Specificity'),
  Straight = sprintf("%.2f ± %.2f",
                     c(mean(accuracy_df$accuracy_One),
                       mean(sensitivity_df$sensitivity_One),
                       mean(specificity_df$specificity_One)),
                     c(sd(accuracy_df$accuracy_One),
                       sd(sensitivity_df$sensitivity_One),
                       sd(specificity_df$specificity_One))
  ),
  Diagonal = sprintf("%.2f ± %.2f",
                     c(mean(accuracy_df$accuracy_Two),
                       mean(sensitivity_df$sensitivity_Two),
                       mean(specificity_df$specificity_Two)),
                     c(sd(accuracy_df$accuracy_Two),
                       sd(sensitivity_df$sensitivity_Two),
                       sd(specificity_df$specificity_Two))
  )
)

summary_table_gt <- summary_table |> 
  gt()
summary_table_gt

