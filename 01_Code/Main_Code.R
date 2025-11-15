#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here(""))

## Additional:

Enable_Catboost <- TRUE

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "dplyr", "tidyr",
              "corrplot",
              "ggplot2", "reshape2", "patchwork",
              "glmnet",
              "randomForest", "gbm",
              "remotes",
              "gridExtra",
              "scorecard")

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

## Optional: Catboost package.
if(Enable_Catboost){
# remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-windows-x86_64-1.2.8.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
library(catboost)
}

#==== 1B - Functions ==========================================================#

## Skewness.
skew <- function(x) {
  x <- na.omit(x)
  n <- length(x)
  mean_x <- mean(x)
  sd_x <- sd(x)
  skewness_value <- (n / ((n - 1) * (n - 2))) * sum(((x - mean_x) / sd_x)^3)
  return(skewness_value)
}

## Use the Macro F1-score as the new loss function.
calculate_macro_f1 <- function(actual, predicted) {
  actual <- as.factor(actual)
  predicted <- as.factor(predicted)
  all_levels <- union(levels(actual), levels(predicted))
  actual <- factor(actual, levels = all_levels)
  predicted <- factor(predicted, levels = all_levels)
  
  cm <- table(Actual = actual, Predicted = predicted)
  
  all_f1_scores <- c()
  for (class in all_levels) {
    tp <- cm[class, class]
    fp <- sum(cm[, class]) - tp
    fn <- sum(cm[class, ]) - tp
    
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    
    f1 <- if ((precision + recall) > 0) 2 * (precision * recall) / (precision + recall) else 0
    all_f1_scores <- c(all_f1_scores, f1)
  }
  
  return(mean(all_f1_scores, na.rm = TRUE))
}

#==== 1C - Parameters =========================================================#

## Directories.
Data_Directory <- file.path(Path, "02_Data")
Charts_Directory <- file.path(Path, "03_Charts")

Dataset_Data_Directory <- file.path(Data_Directory, "wine_data.csv")

## Plotting.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"

height <- 1833
width <- 3750

## Data Sampling.
set.seed(123)

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02a - Read the data file ================================================#
data <- read.csv(Dataset_Data_Directory)
predictors <- data[, -which(names(data) == "quality")]

#==== 02b - Pre-processing and variable selection =============================#

#=======================#
## First let us check if we have any missing data.
#=======================#
colSums(is.na(data)) ## Is not the case, so we can continue.

#=======================#
## Now check the features for potentially different scale (e.g. total_sulfur_diocide is 47 vs 
## chlorides, which is 0.074.)
#=======================#
summary(data)

## We can observe that the data differs considerably. The means and medians
## are off compared to another. 

#=======================#
## Now check our independent variable quality.
#=======================#
table(data$quality) ## We can observe that we have a few, discrete levels (3:9).
                    ## , thus we should change it to a factor (we have a classification task NOT a regression).

# data$quality <- as.numeric(data$quality)
data$quality <- as.factor(data$quality)

#=======================#
## Check for data variability and skewed data.
#=======================#
## Possible bimodial distribution (red vs white wine). Will work well with trees.
hist(data$chlorides)
data$chlorides <- log(data$chlorides)

## Plot the data distribution.
df_long <- data %>%
  select(-quality) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

# Plot histograms with faceting
plot <- ggplot(df_long, aes(x = value)) +
  geom_histogram(fill = blue, color = "white", bins = 30) +
  facet_wrap(~feature, scales = "free", ncol = 3) +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold", size = 10)) +
  labs(title = "",
       x = "Value",
       y = "Count")

Path <- file.path(Charts_Directory, "02_Data_Distribution.png")
ggsave(
  filename = Path,
  plot = plot,
  width = 3750,
  height = 1833,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## log transform.

df_long_log <- df_long
df_long_log$value <- log(df_long_log$value)

# plot <- ggplot(df_long_log, aes(x = value)) +
#   geom_histogram(fill = blue, color = "white", bins = 30) +
#   facet_wrap(~feature, scales = "free", ncol = 3) +
#   theme_minimal() +
#   theme(strip.text = element_text(face = "bold", size = 10)) +
#   labs(title = "",
#        x = "Value",
#        y = "Count")

## Do not log-transform residual_sugar as it will contribute to even more unbalancing.
hist(data$residual_sugar)
# data$residual_sugar <- log(data$residual_sugar)

## Possible bimodial distribution (red vs white wine). Will work well with trees.
hist(data$free_sulfur_dioxide)

## Check the variation of each feature.
Variance <- apply(as.matrix(predictors), MARGIN = 2, FUN = var)
Skewness <- apply(as.matrix(predictors), MARGIN = 2, FUN = skew)

## We should standardize as the variance differs considerably between our features.
## e.g. for the skewness major deviations are chlorides and residual sugar (pointing to a bimodial distribution).
## Given the non-normal distributions, decision tree's might work well.

#=======================#
## Extreme values.
#=======================#
boxplot(data$residual_sugar)


#==== 02c - Multicollinearity and IV ==========================================#

## ======================= ##
## Informational Value.
## ======================= ##
data_iv <- data %>%
  mutate(quality = ifelse(quality >= 6, 1, 0))

## Tells us how good a feature seperates between NoDefault (y=0) and Default (y=1).
iv_summary <- iv(data_iv, y = "quality")
print(iv_summary %>% arrange(desc(info_value)))

## Plot the informational value.

iv_summary <- iv_summary %>%
  mutate(
    power_category = case_when(
      info_value > 0.5   ~ "Very Strong",
      info_value >= 0.3  ~ "Strong",
      info_value >= 0.1  ~ "Medium",
      info_value >= 0.02 ~ "Weak",
      TRUE               ~ "Useless"
    ),
    # Convert to a factor to control the order in the legend
    power_category = factor(power_category, 
                            levels = c("Very Strong", "Strong", "Medium", "Weak", "Useless"))
  )

# 3. Create the ggplot visualization
plot_IV <- ggplot(iv_summary, aes(x = info_value, y = reorder(variable, info_value))) +
  geom_col(aes(fill = power_category)) +
  geom_text(aes(label = round(info_value, 3)), hjust = -0.1, size = 3.5) +
  geom_vline(xintercept = c(0.02, 0.1, 0.3, 0.5), linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.02, y = Inf, label = "Weak", vjust = -0.5, hjust = -0.1, size = 3, color = "gray20") +
  annotate("text", x = 0.1, y = Inf, label = "Medium", vjust = -0.5, hjust = -0.1, size = 3, color = "gray20") +
  annotate("text", x = 0.3, y = Inf, label = "Strong", vjust = -0.5, hjust = -0.1, size = 3, color = "gray20") +
  annotate("text", x = 0.5, y = Inf, label = "Very strong", vjust = -0.5, hjust = -0.1, size = 3, color = "gray20") +
  labs(
    title = "",
    subtitle = "",
    x = "Information Value",
    y = "",
    fill = "Predictive Power"
  ) +
  
  # Manually set colors for the categories
  scale_fill_manual(values = c(
    "Very Strong" = "#d53e4f", 
    "Strong" = "#f46d43", 
    "Medium" = "#fdae61", 
    "Weak" = "#fee08b", 
    "Useless" = "#e6f598"
  )) +
  
  scale_x_continuous(limits = c(0, max(iv_summary$info_value) * 1.1)) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

Path <- file.path(Charts_Directory, "01b_IV_per_feature.png")
ggsave(
  filename = Path,
  plot = plot_IV,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## Multicollinearity.
## ======================= ##

## Parameters.
Path <- file.path(Charts_Directory, "01a_Correlation_Plot.png")

## Main Code.
cor_matrix <- cor(scaled_predictors[,])
cor_matrix_upper_na <- cor_matrix
cor_matrix_upper_na[upper.tri(cor_matrix_upper_na)] <- NA
melted_cor_matrix <- melt(cor_matrix_upper_na, na.rm = TRUE)

## Plot.

plot <- ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") + 
    scale_fill_gradient2(low = blue, high = orange, mid = "white", 
                       midpoint = 0, limit = c(-1, 1), 
                       name = "Correlation") +
    geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
    theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title = element_blank()) + 
  coord_fixed() 

ggsave(
  filename = Path,
  plot = plot,
  width = 3750,
  height = 1833,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## We no longer need a separate 'Validation' set.
## Cross-validation on the 'Train' set will replace it.

## Now we are done preparing the data.
## - Standardized the features (discovered bimodiality in two datasets, points to decision trees).
## - Detected skewness and differences in the variation/variance.
## - Looked at multicollinearity. Most variables show a high correlation with eachother.
## - Dependent variable is set up as a factor. We are working on a classification task.
## - Data is split into Train and Test sets for a robust CV workflow.

#==============================================================================#
#==== 03 - Feature Selection & Engineering and Data preparation ===============#
#==============================================================================#

## ======================= ##
## Feature selection.
## ======================= ##


## ======================= ##
## Feature engineering.
## ======================= ##
## Add a dummy for red and white wines.

features_for_clustering <- c("total_sulfur_dioxide", "chlorides", "volatile_acidity")
data_for_clustering <- data[features_for_clustering]

scaled_data <- scale(data_for_clustering)

# -- Step 3: Apply K-Means clustering --
# We set centers=2 because we are looking for two groups (red and white).
# nstart=25 runs the algorithm 25 times with different starting points to find a stable solution.
# set.seed() makes the result reproducible.
kmeans_result <- kmeans(scaled_data, centers = 2, nstart = 25)
data$cluster <- kmeans_result$cluster
cluster_summary <- data %>%
  group_by(cluster) %>%
  summarise(
    mean_total_sulfur = mean(total_sulfur_dioxide),
    mean_chlorides = mean(chlorides),
    mean_volatile_acidity = mean(volatile_acidity),
    count = n()
  )

print(cluster_summary)

##
white_wine_cluster_id <- cluster_summary %>%
  filter(mean_total_sulfur == max(mean_total_sulfur)) %>%
  pull(cluster)

data <- data %>%
  mutate(is_white = ifelse(cluster == white_wine_cluster_id, 1, 0))
data <- data %>%
  select(-cluster)

print(head(data))
cat("\nCounts of inferred wine types:\n")
print(table(data$is_white))


## ======================= ##
## Standardization.
## ======================= ##

scaled_predictors <- data.frame(scale(predictors))
data_standardized <- cbind(scaled_predictors, data$quality)
colnames(data_standardized) <- colnames(data)

Variance_standardized <- apply(as.matrix(data_standardized), MARGIN = 2, FUN = var)

## ======================= ##
## Data splitting (70/30).
## ======================= ##

train_index <- sample(1:nrow(data_standardized), size = 0.70 * nrow(data_standardized))

Train <- data_standardized[train_index, ]
Test  <- data_standardized[-train_index, ]

## We no longer need a separate 'Validation' set.
## Cross-validation on the 'Train' set will replace it.

## Now we are done preparing the data.
## - Standardized the features (discovered bimodiality in two datasets, points to decision trees).
## - Detected skewness and differences in the variation/variance.
## - Looked at multicollinearity. Most variables show a high correlation with eachother.
## - Dependent variable is set up as a factor. We are working on a classification task.
## - Data is split into Train and Test sets for a robust CV workflow.


#==============================================================================#
#==== 04 - Analysis ===========================================================#
#==============================================================================#

#=======================#
## Parameters.
#=======================#
set.seed(123)
k_folds <- 5

#=======================#
## Main Code.
#=======================#

tryCatch({

## We are now using a k-fold cross-validation workflow.
## For our evaluation metric, we will use F1-score.
## Since the data is perfectly balanced, maximizing accuracy during CV is a 
## direct and efficient proxy for maximizing the F1-score.
  
#==== 03a - Linear regression (Lasso and Ridge) ===============================#
  
## Prepare the data matrices.
train_x <- model.matrix(quality ~ ., data = Train)[, -1]
train_y <- Train$quality
  
test_x <- model.matrix(quality ~ ., data = Test)[, -1]
test_y <- Test$quality

#=======================#
## Lasso.
#=======================#
cv_lasso <- cv.glmnet(train_x, train_y,
                      alpha = 1,
                      family = "multinomial",
                      type.measure = "class", # Minimize misclassification error (maximizes accuracy)
                      nfolds = k_folds)
  
## The best lambda is the one that minimizes the cross-validated error.
best_lambda_lasso <- cv_lasso$lambda.min

#=======================#
## Ridge.
#=======================#
cv_ridge <- cv.glmnet(train_x, train_y,
                      alpha = 0,
                      family = "multinomial",
                      type.measure = "class",
                      nfolds = k_folds)

best_lambda_ridge <- cv_ridge$lambda.min

#=======================#
## Evaluate it on the test set.
#=======================#
## Lasso final evaluation
final_lasso_preds <- predict(cv_lasso, newx = test_x, s = best_lambda_lasso, type = "class")
lasso_test_f1 <- calculate_macro_f1(actual = test_y, predicted = final_lasso_preds)

## Ridge final evaluation
final_ridge_preds <- predict(cv_ridge, newx = test_x, s = best_lambda_ridge, type = "class")
ridge_test_f1 <- calculate_macro_f1(actual = test_y, predicted = final_ridge_preds)

print("--- FINAL MODEL ASSESSMENT (F1-Score on Test Set) ---")
print(paste("Final Lasso F1-Score (Test Set):", round(lasso_test_f1, 4)))
print(paste("Final Ridge F1-Score (Test Set):", round(ridge_test_f1, 4)))

#=======================#
## Visualisation Updates.
#=======================#

plot(cv_lasso)
plot(cv_ridge)

tryCatch({
  
  # We can now update the coefficient plots to show the best lambda found via CV.
  lasso_model_fit <- cv_lasso$glmnet.fit
  ridge_model_fit <- cv_ridge$glmnet.fit
  
  # Lasso Coefficients Plot
  lasso_coefs_matrix <- as.matrix(t(coef(lasso_model_fit)$"6"))
  lasso_df <- as.data.frame(lasso_coefs_matrix)
  lasso_df$lambda <- lasso_model_fit$lambda
  lasso_plot_df <- melt(lasso_df, id.vars = "lambda", variable.name = "Coefficient", value.name = "Value")
  lasso_plot_df <- filter(lasso_plot_df, Coefficient != "(Intercept)")
  
  plot_lasso_coefs <- ggplot(lasso_plot_df, aes(x = lambda, y = Value, color = Coefficient)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = best_lambda_lasso, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_log10() + # Use a log scale for lambda, which is standard for these plots
    labs(title = "Lasso Coefficients (for class '6')",
         subtitle = paste("Optimal lambda found via CV:", round(best_lambda_lasso, 5)),
         x = "Lambda (log scale)", y = "Coefficient Value") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "none")
  
  # Ridge Coefficients Plot
  ridge_coefs_matrix <- as.matrix(t(coef(ridge_model_fit)$"6"))
  ridge_df <- as.data.frame(ridge_coefs_matrix)
  ridge_df$lambda <- ridge_model_fit$lambda
  ridge_plot_df <- melt(ridge_df, id.vars = "lambda", variable.name = "Coefficient", value.name = "Value")
    ridge_plot_df <- filter(ridge_plot_df, Coefficient != "(Intercept)" & Coefficient != "V1")
  
  plot_ridge_coefs <- ggplot(ridge_plot_df, aes(x = lambda, y = Value, color = Coefficient)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = best_lambda_ridge, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_log10() +
    labs(title = "Ridge Coefficients (for class '6')",
         subtitle = paste("Optimal lambda found via CV:", round(best_lambda_ridge, 5)),
         x = "Lambda (log scale)", y = "Coefficient Value") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "bottom",
          legend.title = element_blank())
  
  # Combine and save the plots
  combined_plot <- plot_lasso_coefs | plot_ridge_coefs
  Path <- file.path(Charts_Directory, "03_Linear_Shrinkage_Coefficients_CV.png")
  
  ggsave(
    filename = Path,
    plot = combined_plot,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
}, silent = TRUE)

## End of whole tryCatch() statement.

}, silent = TRUE)

#==== 03b - Decision Tree's - Random Forest ===================================#

tryCatch({
  
#=======================#
## Parameters for CV.
#=======================#
mtry_values <- c(2, 3, 4, 5, 6) ## 2 is the best one.
nfolds <- k_folds

folds <- sample(cut(seq(1, nrow(Train)), breaks = nfolds, labels = FALSE))

#=======================#
## Main Model Tuning with k-fold Cross-Validation.
#=======================#

rf_results <- data.frame(mtry = mtry_values, F1_Score = NA) 
print("--- Starting Random Forest Tuning with 5-fold Cross-Validation ---")

for (i in 1:length(mtry_values)) {
  
  current_mtry <- mtry_values[i]
  fold_f1_scores <- c() # Store F1 scores for each fold
  
  for (k in 1:nfolds) {
    
    val_indices <- which(folds == k)
    train_fold <- Train[-val_indices, ]
    val_fold   <- Train[val_indices, ]
    
    rf_model <- randomForest(
      quality ~ ., 
      data = train_fold, 
      mtry = current_mtry,
      ntree = 500
    )
    
    val_preds <- predict(rf_model, newdata = val_fold)
    f1 <- calculate_macro_f1(actual = val_fold$quality, predicted = val_preds)
    fold_f1_scores <- c(fold_f1_scores, f1)
  }
  
  mean_f1 <- mean(fold_f1_scores)
  rf_results$F1_Score[i] <- mean_f1
  
  print(paste("Completed mtry =", current_mtry, 
              "| Mean CV F1-Score:", round(mean_f1, 4)))
}

print("--- Tuning Complete ---")
print(rf_results)

best_mtry <- rf_results$mtry[which.max(rf_results$F1_Score)] 
print(paste("Best mtry value found (max F1-Score):", best_mtry))

#=======================#
## Final Model Training and Assessment.
#=======================#
final_rf_model <- randomForest(
  quality ~ ., 
  data = Train, # Using the full 80% training set
  mtry = best_mtry,
  ntree = 500,
  importance = TRUE
)

rf_test_preds <- predict(final_rf_model, newdata = Test)
rf_final_f1 <- calculate_macro_f1(actual = Test$quality, predicted = rf_test_preds)

print("--- FINAL RANDOM FOREST ASSESSMENT ---")
print(paste("Final RF F1-Score (Test Set):", round(rf_final_f1, 4)))

#=======================#
## Visualisation.
#=======================#

tryCatch({
  
  #--- Plot 1: Feature Importance (Your Original Plot) ---#
  importance_data <- as.data.frame(importance(final_rf_model))
  importance_data$Variable <- rownames(importance_data)
  
  plot_importance <- ggplot(importance_data, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
    geom_bar(stat = "identity", fill = blue) +
    coord_flip() +
    labs(title = "Feature Importance (Random Forest Classification)",
         subtitle = paste("Best mtry =", best_mtry),
         x = "Features",
         y = "Mean Decrease in Accuracy") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04_Random_Forest_FeatureImportance_Class.png")
  ggsave(
    filename = Path,
    plot = plot_importance,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  #--- Plot 2: Raw Count Confusion Matrix (Your Original Plot) ---#
  cm_data_raw <- as.data.frame(table(Actual = Test$quality, Predicted = rf_test_preds))
  
  plot_cm_raw <- ggplot(cm_data_raw, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = blue) +
    geom_text(aes(label = Freq), vjust = 1) +
    labs(title = "Random Forest Confusion Matrix (Raw Counts)",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Frequency") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04b_Random_Forest_ConfusionMatrix.png")
  ggsave(
    filename = Path,
    plot = plot_cm_raw,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  #--- Plot 3: Normalized Confusion Matrix (NEW) ---#
  
  ## Calculate percentages for the plot
  cm_percent_rf <- cm_data_raw %>%
    group_by(Actual) %>%
    mutate(Percentage = Freq / sum(Freq),
           Label = paste0(round(Percentage * 100), "%"))
  
  ## Create the heatmap plot
  plot_cm_normalized <- ggplot(cm_percent_rf, aes(x = Actual, y = Predicted, fill = Percentage)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = orange, labels = scales::percent) +
    geom_text(aes(label = Label), vjust = 1) +
    labs(title = "Normalized Confusion Matrix (Random Forest)",
         subtitle = "Rows sum to 100%",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Percentage of Actual") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          aspect.ratio = 1)
  
  ## Save the plot
  Path <- file.path(Charts_Directory, "04e_Random_Forest_Normalized_CM.png")
  ggsave(
    filename = Path,
    plot = plot_cm_normalized,
    width = 2500,
    height = 2500,
    units = "px",
    dpi = 300
  )
  
  #--- Plot 4: Error Analysis Plot (NEW) ---#
  
  ## Create a dataframe of test results, including the predictions
  results_df <- Test
  results_df$Predicted <- rf_test_preds
  results_df$Status <- ifelse(results_df$quality == results_df$Predicted, "Correct", "Incorrect")
  
  ## Filter for only the misclassified samples
  misclassified_df <- results_df %>%
    filter(Status == "Incorrect")
  
  ## Create a plot showing where the errors occur
  plot_error_analysis <- ggplot(misclassified_df, aes(x = quality, y = Predicted)) +
    geom_count(aes(color = after_stat(n)), show.legend = TRUE) +
    scale_color_gradient(low = blue, high = red) +
    labs(title = "Analysis of Misclassifications (Random Forest)",
         subtitle = "Where are the model's errors concentrated?",
         x = "Actual Quality",
         y = "Predicted Quality",
         size = "Number of Errors",
         color = "Count") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          aspect.ratio = 1) +
    guides(color = "none")
  
  ## Save the plot
  Path <- file.path(Charts_Directory, "04f_Random_Forest_Error_Analysis.png")
  ggsave(
    filename = Path,
    plot = plot_error_analysis,
    width = 2500,
    height = 2500,
    units = "px",
    dpi = 300
  )

## End of the tryCatch() statement.

}, silent = TRUE)

#==== 03c - Decision Tree's - Gradient Boosting ===============================#

tryCatch({

#=======================#
## Parameters & Data Prep.
#=======================#
## The gbm package requires the target variable to be 0-indexed for multinomial classification.
## We will create copies of our dataframes for this purpose.
  
Train_GBM <- Train
Train_GBM$quality <- as.integer(Train_GBM$quality) - 3 # Levels 3-9 become 0-6
  
Test_GBM <- Test
Test_GBM$quality <- as.integer(Test_GBM$quality) - 3 # Levels 3-9 become 0-6
  
depth_values <- c(4, 6, 8, 10)
shrinkage <- 0.01

#=======================#
## Main Model Tuning with Built-in Cross-Validation.
#=======================#
gbm_results <- data.frame(depth = depth_values, 
                          best_trees = NA, 
                          cv_Error = NA) 

print("--- Starting GBM Tuning with 5-fold Cross-Validation ---")

for (i in 1:length(depth_values)) {
  gbm_model <- gbm(
    quality ~ .,
    data = Train_GBM, # NOTE: We now use our 80% Train set directly
    distribution = "multinomial",
    n.trees = 2000,
    interaction.depth = depth_values[i],
    shrinkage = shrinkage,
    cv.folds = 5, # The model performs k-fold CV internally
    n.minobsinnode = 10,      
    n.cores = NULL 
  )
  
  best_M <- gbm.perf(gbm_model, plot.it = FALSE, method = "cv")
  best_cv_error <- gbm_model$cv.error[best_M] 
  
  gbm_results$cv_Error[i] <- best_cv_error
  gbm_results$best_trees[i] <- best_M
  
  print(paste("Depth:", depth_values[i], "| Best Trees:", best_M, 
              "| Min CV Error (Deviance):", round(best_cv_error, 4)))
}

print("--- Tuning Complete ---")
print(gbm_results)

best_depth <- gbm_results$depth[which.min(gbm_results$cv_Error)]
best_trees <- gbm_results$best_trees[which.min(gbm_results$cv_Error)]

print(paste("Best Interaction Depth found:", best_depth))
print(paste("Best Number of Trees found:", best_trees))

#=======================#
## Final Model Training and Assessment.
#=======================#
final_gbm_model <- gbm(
  quality ~ .,
  data = Train_GBM, # Using the full 80% training set
  distribution = "multinomial", 
  n.trees = best_trees,
  interaction.depth = best_depth,
  shrinkage = shrinkage,
  n.minobsinnode = 10,
  verbose = FALSE,
  n.cores = NULL
)

gbm_pred_probs <- predict(final_gbm_model, 
                          newdata = Test_GBM, 
                          n.trees = best_trees,
                          type = "response") 

gbm_prob_matrix <- gbm_pred_probs[,,1]
gbm_final_preds_numeric <- apply(gbm_prob_matrix, 1, which.max) - 1

gbm_final_preds_factor <- as.factor(gbm_final_preds_numeric + 3)

gbm_final_f1 <- calculate_macro_f1(actual = Test$quality, predicted = gbm_final_preds_factor)

print("--- FINAL GRADIENT BOOSTING ASSESSMENT ---")
print(paste("Final GBM F1-Score (Test Set):", round(gbm_final_f1, 4)))

#=======================#
## Visualisation.
#=======================#

tryCatch({
  
  #--- Plot 1: Feature Importance (Your Original Plot) ---#
  importance_data_gbm <- as.data.frame(summary(final_gbm_model))
  
  plot_importance_gbm <- ggplot(importance_data_gbm, aes(x = reorder(var, rel.inf), y = rel.inf)) +
    geom_bar(stat = "identity", fill = blue) +
    coord_flip() +
    labs(title = "Feature Importance (GBM Classification)",
         subtitle = paste("Best depth =", best_depth, "& Best n.trees =", best_trees),
         x = "Features",
         y = "Relative Influence") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04c_GBM_FeatureImportance.png")
  ggsave(
    filename = Path,
    plot = plot_importance_gbm,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  #--- Plot 2: Raw Count Confusion Matrix ---#
  actual_labels    <- Test$quality
  predicted_labels <- gbm_final_preds_factor
  
  # Ensure all factor levels are present for a complete matrix
  all_levels <- levels(Test$quality)
  actual_labels <- factor(actual_labels, levels = all_levels)
  predicted_labels <- factor(predicted_labels, levels = all_levels)
  
  cm_data_raw_gbm <- as.data.frame(table(Actual = actual_labels, Predicted = predicted_labels))
  
  plot_cm_raw_gbm <- ggplot(cm_data_raw_gbm, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = blue) +
    geom_text(aes(label = Freq), vjust = 1) +
    labs(title = "GBM Confusion Matrix (Raw Counts)",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Frequency") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04d_GBM_ConfusionMatrix.png")
  ggsave(
    filename = Path,
    plot = plot_cm_raw_gbm,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  #--- Plot 3: Normalized Confusion Matrix (NEW) ---#
  
  cm_percent_gbm <- cm_data_raw_gbm %>%
    group_by(Actual) %>%
    mutate(Percentage = Freq / sum(Freq),
           Label = paste0(round(Percentage * 100), "%"))
  
  plot_cm_normalized_gbm <- ggplot(cm_percent_gbm, aes(x = Actual, y = Predicted, fill = Percentage)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = orange, labels = scales::percent) +
    geom_text(aes(label = Label), vjust = 1) +
    labs(title = "Normalized Confusion Matrix (GBM)",
         subtitle = "Rows sum to 100%",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Percentage of Actual") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          aspect.ratio = 1)
  
  ## Save the plot
  Path <- file.path(Charts_Directory, "04g_GBM_Normalized_CM.png")
  ggsave(
    filename = Path,
    plot = plot_cm_normalized_gbm,
    width = 2500,
    height = 2500,
    units = "px",
    dpi = 300
  )
  
  #--- Plot 4: Error Analysis Plot (NEW) ---#
  
  results_df_gbm <- Test
  results_df_gbm$Predicted <- gbm_final_preds_factor
  results_df_gbm$Status <- ifelse(results_df_gbm$quality == results_df_gbm$Predicted, "Correct", "Incorrect")
  
  misclassified_df_gbm <- results_df_gbm %>%
    filter(Status == "Incorrect")
  
  ## Create a plot showing where the errors occur
  plot_error_analysis_gbm <- ggplot(misclassified_df_gbm, aes(x = quality, y = Predicted)) +
    geom_count(aes(color = after_stat(n)), show.legend = TRUE) +
    scale_color_gradient(low = blue, high = red) +
    labs(title = "Analysis of Misclassifications (GBM)",
         subtitle = "Where are the model's errors concentrated?",
         x = "Actual Quality",
         y = "Predicted Quality",
         size = "Number of Errors",
         color = "Count") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          aspect.ratio = 1) +
    guides(color = "none")
  
  ## Save the plot
  Path <- file.path(Charts_Directory, "04h_GBM_Error_Analysis.png")
  ggsave(
    filename = Path,
    plot = plot_error_analysis_gbm,
    width = 2500,
    height = 2500,
    units = "px",
    dpi = 300
  )
  
}, silent = TRUE)

}, silent = TRUE)

#==============================================================================#
#==== 04 - Model Comparison ===================================================#
#==============================================================================#

#==== 04a - Data ==============================================================#

results_df <- data.frame(
  Model = c("Lasso", "Ridge", "Random Forest", "Gradient Boosting"),
  F1_Score = c(lasso_test_f1, 
               ridge_test_f1, 
               rf_final_f1, 
               gbm_final_f1)
)

print("--- Final Model Performance ---")
results_df <- results_df[order(results_df$F1_Score, decreasing = TRUE), ]
print(results_df)


#==== 04b - Visualisation =====================================================#

#=====================#
## F1-Score Bar Chart.
#=====================#

plot_f1_comparison <- ggplot(results_df, aes(x = reorder(Model, F1_Score), y = F1_Score)) +
  geom_bar(stat = "identity", aes(fill = Model), width = 0.7) +
  geom_text(aes(label = round(F1_Score, 3)), vjust = -0.5, size = 4) +
  labs(title = "Final Model Comparison: F1-Score on Test Set",
       x = "Model",
       y = "Macro F1-Score") +
  ylim(0, 1) + # F1-score is always between 0 and 1
  theme_light() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1)) # Angle labels if they overlap

Path <- file.path(Charts_Directory, "05_F1_Score_Comparison.png")
ggsave(
  filename = Path,
  plot = plot_f1_comparison,
  width = 2500,
  height = 1800,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

#==== 04c - Advanced Visualisation (Lollipop Chart + Table) ==============#

# Create the Lollipop Chart
# This is a cleaner alternative to a bar chart
plot_lollipop <- ggplot(results_df, aes(x = reorder(Model, F1_Score), y = F1_Score)) +
  geom_segment(aes(xend = reorder(Model, F1_Score), yend = 0), color = grey) +
  geom_point(aes(color = Model), size = 5, show.legend = FALSE) +
  geom_text(aes(label = round(F1_Score, 4)), hjust = -0.4, size = 3.5) +
  coord_flip() + # Flips the chart to be horizontal, which is easier to read
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0.01)) +
  labs(title = "Final Model Performance Ranking",
       subtitle = "Macro F1-Score on the 20% Test Set",
       x = "",
       y = "Macro F1-Score") +
  theme_light() +
  theme(
    panel.border = element_blank(),
    panel.grid.major.y = element_blank(), # Remove horizontal grid lines
    axis.ticks.y = element_blank() # Remove y-axis ticks
  )

results_df_formatted <- results_df
results_df_formatted$F1_Score <- round(results_df_formatted$F1_Score, 4)
table_grob <- tableGrob(results_df_formatted, rows = NULL, theme = ttheme_minimal())

plot_combined <- grid.arrange(plot_lollipop, table_grob, 
                              ncol = 1, 
                              heights = c(2.5, 1), # Give more space to the plot
                              top = "Overall Model Comparison")

Path <- file.path(Charts_Directory, "06_Combined_Comparison.png")
ggsave(
  filename = Path,
  plot = plot_combined,
  width = 2500,
  height = 1800,
  units = "px",
  dpi = 300
)

#==============================================================================#
#==== 05 - Appendix ===========================================================#
#==============================================================================#

#==== 05a - Implement the CatBoost Algorithm ==================================#

tryCatch({
    
#=======================#
## Parameters & CV Setup.
#=======================#
param_grid <- expand.grid(
learning_rate = c(0.03, 0.1),
depth = c(10, 12), # Reduced for speed, you can expand this back if needed
l2_leaf_reg = c(1, 5) )

nfolds <- k_folds
    
## Create the fold indices from the main Train set
folds <- sample(cut(seq(1, nrow(Train)), breaks = nfolds, labels = FALSE))
    
## Prepare the final Test pool (can be done once)
test_x <- Test[, -which(names(Test) == "quality")]
test_y <- as.numeric(Test$quality) - 3 # CatBoost needs 0-indexed numeric labels
test_pool <- catboost.load_pool(data = test_x, label = test_y)
    
## Setup results dataframe
tuning_results <- param_grid
tuning_results$mean_F1 <- NA
tuning_results$best_iteration <- NA
    
#=======================#
## Main Model Tuning with k-fold Cross-Validation.
#=======================#  
    print("--- STARTING CATBOOST TUNING WITH 5-FOLD CROSS-VALIDATION ---")
    
    for (i in 1:nrow(param_grid)) {
      
      fold_f1_scores <- c()
      fold_iterations <- c()
      
      for (k in 1:nfolds) {
        val_indices <- which(folds == k)
        train_fold <- Train[-val_indices, ]
        val_fold   <- Train[val_indices, ]
        
        # Create data pools for this specific fold
        train_fold_x <- train_fold[, -which(names(train_fold) == "quality")]
        train_fold_y <- as.numeric(train_fold$quality) - 3
        val_fold_x <- val_fold[, -which(names(val_fold) == "quality")]
        val_fold_y <- as.numeric(val_fold$quality) - 3
        
        train_pool_fold <- catboost.load_pool(train_fold_x, label = train_fold_y)
        val_pool_fold <- catboost.load_pool(val_fold_x, label = val_fold_y)
        
        params <- list(
          iterations = 2000,
          loss_function = 'MultiClass',
          eval_metric = 'TotalF1', # Using F1-score for tuning
          use_best_model = TRUE,
          early_stopping_rounds = 50,
          learning_rate = param_grid$learning_rate[i],
          depth = param_grid$depth[i],
          l2_leaf_reg = param_grid$l2_leaf_reg[i],
          logging_level = 'Silent',
          train_dir = "catboost_logs" # Use a dedicated log dir
        )
        
        model_fold <- catboost.train(learn_pool = train_pool_fold, test_pool = val_pool_fold, params = params)
        
        run_metrics <- read.delim(file.path("catboost_logs", "test_error.tsv"))
        best_idx <- which.max(run_metrics$TotalF1)
        fold_f1_scores <- c(fold_f1_scores, run_metrics$TotalF1[best_idx])
        fold_iterations <- c(fold_iterations, run_metrics$iter[best_idx] + 1)
      }
      
      tuning_results$mean_F1[i] <- mean(fold_f1_scores)
      tuning_results$best_iteration[i] <- ceiling(mean(fold_iterations)) # Use the average best iteration
      
      print(paste("Completed run", i, "/", nrow(param_grid), 
                  "| Avg Best Iter:", tuning_results$best_iteration[i],
                  "| Mean CV F1-Score:", round(tuning_results$mean_F1[i], 5)))
    }
    
    print("--- TUNING COMPLETE ---")
    print(tuning_results)
    
#=======================#
## Final Model Training & Assessment.
#=======================#  
    best_params_row <- tuning_results[which.max(tuning_results$mean_F1), ]
    
    print("--- BEST HYPERPARAMETERS FOUND ---")
    print(best_params_row)
    
    final_train_x <- Train[, -which(names(Train) == "quality")]
    final_train_y <- as.numeric(Train$quality) - 3
    final_train_pool <- catboost.load_pool(data = final_train_x, label = final_train_y)
    
    final_params <- list(
      iterations = best_params_row$best_iteration, 
      loss_function = 'MultiClass',
      learning_rate = best_params_row$learning_rate,
      depth = best_params_row$depth,
      l2_leaf_reg = best_params_row$l2_leaf_reg,
      logging_level = 'Silent'
    )
    
    print("--- TRAINING FINAL MODEL ON FULL TRAIN SET ---")
    final_catboost_model <- catboost.train(learn_pool = final_train_pool, params = final_params)
    
## Evaluate the final model on the unseen Test set
    catboost_preds_numeric <- catboost.predict(final_catboost_model, test_pool)
    catboost_preds_factor <- as.factor(catboost_preds_numeric + 3)
    
    catboost_final_f1 <- calculate_macro_f1(actual = Test$quality, predicted = catboost_preds_factor)
    
    print("--- FINAL CATBOOST ASSESSMENT ---")
    print(paste("Final CatBoost F1-Score (Test Set):", round(catboost_final_f1, 5)))
    
#=======================#
## Visualization.
#=======================#    

#--- Plot 1: Feature Importance ---#
    tryCatch({
      importance_data_cb <- as.data.frame(catboost.get_feature_importance(final_catboost_model))
      importance_data_cb$Variable <- rownames(importance_data_cb)
      colnames(importance_data_cb)[1] <- "Importance"
      
      plot_cb_imp <- ggplot(importance_data_cb, aes(x = reorder(Variable, Importance), y = Importance)) +
        geom_bar(stat = "identity", fill = blue) +
        coord_flip() +
        labs(title = "Feature Importance (CatBoost)",
             x = "Features", y = "Importance") +
        theme_light() +
        theme(plot.title = element_text(hjust = 0.5))
      
      Path <- file.path(Charts_Directory, "04i_CatBoost_FeatureImportance.png")
      ggsave(filename = Path, plot = plot_cb_imp, width = 3750, height = 1833, units = "px", dpi = 300)
    }, silent = TRUE)
    
    #--- Plot 2 & 3: Confusion Matrices (Raw and Normalized) ---#
    tryCatch({
      cm_data_raw_cb <- as.data.frame(table(Actual = Test$quality, Predicted = catboost_preds_factor))
      
      # Normalized CM
      cm_percent_cb <- cm_data_raw_cb %>% group_by(Actual) %>% mutate(Percentage = Freq / sum(Freq), Label = paste0(round(Percentage * 100), "%"))
      plot_cm_normalized_cb <- ggplot(cm_percent_cb, aes(x = Actual, y = Predicted, fill = Percentage)) +
        geom_tile(color = "white") +
        scale_fill_gradient(low = "white", high = orange, labels = scales::percent) +
        geom_text(aes(label = Label), vjust = 1) +
        labs(title = "Normalized Confusion Matrix (CatBoost)", subtitle = "Rows sum to 100%", x = "Actual Quality", y = "Predicted Quality", fill = "Percentage") +
        theme_light() +
        theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5), aspect.ratio = 1)
      
      Path <- file.path(Charts_Directory, "04j_CatBoost_Normalized_CM.png")
      ggsave(filename = Path, plot = plot_cm_normalized_cb, width = 2500, height = 2500, units = "px", dpi = 300)
    }, silent = TRUE)
    
    #--- Plot 4: Error Analysis ---#
    tryCatch({
      results_df_cb <- Test
      results_df_cb$Predicted <- catboost_preds_factor
      misclassified_df_cb <- results_df_cb %>% filter(quality != Predicted)
      
      plot_error_analysis_cb <- ggplot(misclassified_df_cb, aes(x = quality, y = Predicted)) +
        geom_count(aes(color = after_stat(n))) +
        scale_color_gradient(low = blue, high = red) +
        labs(title = "Analysis of Misclassifications (CatBoost)", subtitle = "Where are the model's errors concentrated?", x = "Actual Quality", y = "Predicted Quality", size = "Number of Errors") +
        theme_light() +
        theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5), aspect.ratio = 1) +
        guides(color = "none")
      
      Path <- file.path(Charts_Directory, "04k_CatBoost_Error_Analysis.png")
      ggsave(filename = Path, plot = plot_error_analysis_cb, width = 2500, height = 2500, units = "px", dpi = 300)
    }, silent = TRUE)
    
  }, silent = TRUE)
  

#==============================================================================#
#==============================================================================#
#==============================================================================#