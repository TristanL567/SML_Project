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
              "remotes")

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

#=======================#
## Standardize the features.
#=======================#
scaled_predictors <- data.frame(scale(predictors))
data_standardized <- cbind(scaled_predictors, data$quality)
colnames(data_standardized) <- colnames(data)

Variance_standardized <- apply(as.matrix(data_standardized), MARGIN = 2, FUN = var)

#==== 02c - Multicollinearity =================================================#

## Parameters.
Path <- file.path(Charts_Directory, "01_Correlation_Plot.png")

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

#==== 02d - Splitting the dataset =============================================#
## 50/25/25 rule - train, validation, test
set.seed(123)

train_index <- sample(1:nrow(data_standardized), size = 0.75 * nrow(data_standardized))
Train <- data_standardized[train_index, ]
Test  <- data_standardized[-train_index, ]

validation_index <- sample(1:nrow(Train), size = (2.5/7.5) * nrow(Train))
Validation <- Train[validation_index, ]   
Train  <- Train[-validation_index, ]

## Now we are done preparing the data.
## - Standardized the features (discovered bimodiality in two datasets, points to decision trees).
## - Detected skewness and differences in the variation/variance.
## - Looked at multicollinearity. Most variables show a high correlation with eachother.
## - Dependent variable is set up as a factor. We are working on a classification task.

#==============================================================================#
#==== 03 - Analysis ===========================================================#
#==============================================================================#

set.seed(123)

tryCatch({

## Generally, we rely on RMSE is our loss function. It is unreliable
## if the dependent variable is very unbalanced, but we are lucky:
## the dataset is very balanced and that the data is ordinal!

#==== 03a - Linear regression (Lasso and Ridge) ===============================#
train_x <- model.matrix(quality ~ ., data = Train)[, -1]
train_y <- Train$quality

val_x <- model.matrix(quality ~ ., data = Validation)[, -1]
val_y <- Validation$quality

test_x <- model.matrix(quality ~ ., data = Test)[, -1]
test_y <- Test$quality

#=======================#
## Lasso.
#=======================#
# Change family to "gaussian" for regression
lasso_model <- glmnet(train_x, train_y, 
                      alpha = 1, 
                      family = "multinomial")
lambdas <- lasso_model$lambda

val_preds <- predict(lasso_model, newx = val_x, s = lambdas, type = "class")
accuracies <- rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
  accuracies[i] <- mean(val_preds[, i] == val_y)
}

best_lasso_acc <- max(accuracies)
best_lambda <- lambdas[which.max(accuracies)]

#=======================#
## Ridge.
#=======================#
ridge_model <- glmnet(train_x, train_y, 
                      alpha = 0, 
                      family = "multinomial")

lambdas_r <- ridge_model$lambda
val_preds_r <- predict(ridge_model, newx = val_x, s = lambdas_r, type = "class")

accuracies_r <- rep(NA, length(lambdas_r))
for (i in 1:length(lambdas_r)) {
  accuracies_r[i] <- mean(val_preds_r[, i] == val_y)
}

best_ridge_acc <- max(accuracies_r)
best_lambda_r <- lambdas_r[which.max(accuracies_r)]

#=======================#
## Evaluate it on the test set.
#=======================#
final_lasso_preds <- predict(lasso_model, newx = test_x, s = best_lambda, type = "class")
final_ridge_preds <- predict(ridge_model, newx = test_x, s = best_lambda_r, type = "class")

lasso_acc <- mean(final_lasso_preds == test_y)
ridge_acc <- mean(final_ridge_preds == test_y)

lasso_test_rmse <- NA
ridge_test_rmse <- NA

print("--- FINAL MODEL ASSESSMENT (Classification Approach) ---")
print(paste("Final Lasso Accuracy (Test Set):", round(lasso_acc, 4)))
print(paste("Final Ridge Accuracy (Test Set):", round(ridge_acc, 4)))

#=======================#
## Visualisation Updates.
#=======================#

tryCatch({
  
  # Changed to plot Accuracy
  lasso_acc_df <- data.frame(lambda = lambdas, accuracy = accuracies)
  ridge_acc_df <- data.frame(lambda = lambdas_r, accuracy = accuracies_r)
  
  plot_lasso_acc <- ggplot(lasso_acc_df, aes(x = lambda, y = accuracy)) +
    geom_line(linewidth = 1, color = "#7B68EE") +
    geom_vline(xintercept = best_lambda, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_reverse() +
    labs(title = "Lasso: Validation Accuracy vs. Lambda", x = "Lambda", y = "Validation Accuracy") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot_ridge_acc <- ggplot(ridge_acc_df, aes(x = lambda, y = accuracy)) +
    geom_line(linewidth = 1, color = "#CD5C5C") +
    geom_vline(xintercept = best_lambda_r, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_reverse() +
    labs(title = "Ridge: Validation Accuracy vs. Lambda", x = "Lambda", y = "Validation Accuracy") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # You can plot these if you want
  # print(plot_lasso_acc)
  # print(plot_ridge_acc)
  
}, silent = TRUE)

tryCatch({
  
  lasso_coefs_matrix <- as.matrix(t(coef(lasso_model)$"6"))
  lasso_df <- as.data.frame(lasso_coefs_matrix)
  lasso_df$lambda <- lasso_model$lambda
  
  lasso_plot_df <- melt(lasso_df, id.vars = "lambda", variable.name = "Coefficient", value.name = "Value")
  lasso_plot_df <- filter(lasso_plot_df, Coefficient != "(Intercept)")
  
  lasso_plot <- ggplot(lasso_plot_df, aes(x = lambda, y = Value, color = Coefficient)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = best_lambda, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_reverse() +
    labs(title = "Lasso Coefficients (for class '6')", 
         subtitle = paste("Optimal lambda:", round(best_lambda, 5)), 
         x = "Lambda", y = "Coefficient") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5), 
          plot.subtitle = element_text(hjust = 0.5), 
          legend.position = "none",
          legend.title = element_blank())
  
  # Ridge Coefs for class "6"
  ridge_coefs_matrix <- as.matrix(t(coef(ridge_model)$"6")) 
  ridge_df <- as.data.frame(ridge_coefs_matrix)
  ridge_df$lambda <- ridge_model$lambda
  
  ridge_plot_df <- melt(ridge_df, id.vars = "lambda", variable.name = "Coefficient", value.name = "Value")
  ridge_plot_df <- filter(ridge_plot_df, Coefficient != "(Intercept)")
  
  ridge_plot <- ggplot(
    filter(ridge_plot_df, Coefficient != "V1" & Coefficient != "(Intercept)"), 
    aes(x = lambda, y = Value, color = Coefficient)
  ) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = best_lambda_r, linetype = "dashed", color = "black", linewidth = 1) +
    scale_x_reverse() +
    coord_cartesian(xlim = c(2.5, 0)) +
    labs(title = "Ridge Coefficients (for class '6')", 
         subtitle = paste("Optimal lambda:", round(best_lambda_r, 5)), 
         x = "Lambda", y = "Coefficient") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5), 
          plot.subtitle = element_text(hjust = 0.5), 
          legend.position = "bottom", 
          legend.title = element_blank(),
          legend.text = element_text(size = 10))
  
  # Save or display
  
  plot <- ridge_plot | lasso_plot
  Path <- file.path(Charts_Directory, "03_Linear_Shrinkage_Coefficients.png")
  
  ggsave(
    filename = Path,
    plot = plot,
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

set.seed(123)

tryCatch({
  
#=======================#
## Parameters.
#=======================#

mtry_values <- c(2, 3, 4, 5, 6)

#=======================#
## Main Model.
#=======================#

Train$quality <- as.factor(Train$quality)
Validation$quality <- as.factor(Validation$quality)
Test$quality <- as.factor(Test$quality)

rf_results <- data.frame(mtry = mtry_values, Accuracy = NA) 

for (i in 1:length(mtry_values)) {
  
  rf_model <- randomForest(
    quality ~ ., 
    data = Train, 
    mtry = mtry_values[i],
    ntree = 500  
  )
  
  val_preds <- predict(rf_model, newdata = Validation)
    rf_results$Accuracy[i] <- mean(val_preds == Validation$quality)
  
  print(paste("Completed mtry =", mtry_values[i], 
              "| Validation Accuracy:", round(rf_results$Accuracy[i], 4)))
}

print("--- Tuning Complete ---")
print(rf_results)

best_mtry <- rf_results$mtry[which.max(rf_results$Accuracy)] 
print(paste("Best mtry value found (max accuracy):", best_mtry))

#=======================#
## Model Assessment on the training data.
#=======================#

final_rf_model <- randomForest(
  quality ~ ., 
  data = Train, 
  mtry = best_mtry,
  ntree = 500,
  importance = TRUE
)

rf_test_preds <- predict(final_rf_model, newdata = Test)

rf_final_acc <- mean(rf_test_preds == Test$quality)
rf_final_rmse <- NA 

print("--- FINAL RANDOM FOREST ASSESSMENT ---")
print(paste("Final RF Accuracy (Test Set):", rf_final_acc))

#=======================#
## Visualisation.
#=======================#

tryCatch({
  
  importance_data <- as.data.frame(importance(final_rf_model))
  importance_data$Variable <- rownames(importance_data)

  plot <- ggplot(importance_data, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
    geom_bar(stat = "identity", fill = blue) +
    coord_flip() +
    labs(title = "Feature Importance (Random Forest Classification)",
         subtitle = "",
         x = "Features",
         y = "Mean Decrease in Accuracy") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04_Random_Forest_FeatureImportance_Class.png")
  
  ggsave(
    filename = Path,
    plot = plot,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  cm_data <- as.data.frame(table(Actual = Test$quality, Predicted = rf_test_preds))
  
  plot_cm <- ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = blue) +
    geom_text(aes(label = Freq), vjust = 1) +
    labs(title = "Random Forest Confusion Matrix",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Frequency") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04b_Random_Forest_ConfusionMatrix.png")
  
  ggsave(
    filename = Path,
    plot = plot_cm,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
  
}, silent = TRUE)

## End of the tryCatch() statement.

}, silent = TRUE)

#==== 03c - Decision Tree's - Gradient Boosting ===============================#

set.seed(123)

tryCatch({

#=======================#
## Parameters.
#=======================#

  Train_GBM <- Train
  Validation_GBM <- Validation
  Test_GBM <- Test
  
  Train_GBM$quality <- as.integer(Train$quality) - 1
  Validation_GBM$quality <- as.integer(Validation$quality) - 1
  Test_GBM$quality <- as.integer(Test$quality) - 1
  
  Train_Num <- rbind(Train_GBM, Validation_GBM)
  
depth_values <- c(6, 8, 10)
shrinkage <- 0.01

#=======================#
## Main Model.
#=======================#
gbm_results <- data.frame(depth = depth_values, 
                          best_trees = NA, 
                          cv_Error = NA) 

print("--- Starting GBM Tuning (Classification Mode) ---")

for (i in 1:length(depth_values)) {
  gbm_model <- gbm(
    quality ~ .,
    data = Train_Num,
    distribution = "multinomial",
    n.trees = 2000,
    interaction.depth = depth_values[i],
    shrinkage = shrinkage,
    cv.folds = 5,
    n.minobsinnode = 10,      
    n.cores = NULL 
  )
  
  best_M <- gbm.perf(gbm_model, plot.it = FALSE, method = "cv")
    best_cv_error <- gbm_model$cv.error[best_M] 
  
  gbm_results$cv_Error[i] <- best_cv_error
  gbm_results$best_trees[i] <- best_M
  
  print(paste("Depth:", depth_values[i], "| Best Trees:", best_M, 
              "| CV Error (Deviance):", round(best_cv_error, 4)))
}

print("--- Tuning Complete ---")
print(gbm_results)

best_depth <- gbm_results$depth[which.min(gbm_results$cv_Error)]
best_trees <- gbm_results$best_trees[which.min(gbm_results$cv_Error)]

print(paste("Best Interaction Depth (J) found:", best_depth))
print(paste("Best Number of Trees (M) found:", best_trees))

#=======================#
## Final Model.
#=======================#
final_gbm_model <- gbm(
  quality ~ .,
  data = Train_Num, 
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
gbm_final_preds <- apply(gbm_prob_matrix, 1, which.max) - 1

gbm_final_acc <- mean(gbm_final_preds == Test_GBM$quality)
gbm_final_rmse <- NA 

print("--- FINAL GRADIENT BOOSTING ASSESSMENT ---")
print(paste("Final GBM Accuracy (Test Set):", round(gbm_final_acc, 4)))

#=======================#
## Visualisation.
#=======================#

tryCatch({
  
  importance_data_gbm <- as.data.frame(summary(final_gbm_model))
  
  plot_gbm_imp <- ggplot(importance_data_gbm, aes(x = reorder(var, rel.inf), y = rel.inf)) +
    geom_bar(stat = "identity", fill = blue) +
    coord_flip() +
    labs(title = "Feature Importance (GBM Classification)",
         x = "Features",
         y = "Relative Influence") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04c_GBM_FeatureImportance.png")
  
  ggsave(
    filename = Path,
    plot = plot_gbm_imp,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
}, silent = TRUE)


tryCatch({
  
  actual_labels    <- as.factor(Test_GBM$quality + 3)
  predicted_labels <- as.factor(gbm_final_preds + 3)
  
  all_levels <- as.factor(3:9)
  actual_labels <- factor(actual_labels, levels = levels(all_levels))
  predicted_labels <- factor(predicted_labels, levels = levels(all_levels))
  
  cm_data_gbm <- as.data.frame(table(Actual = actual_labels, Predicted = predicted_labels))
  
  plot_gbm_cm <- ggplot(cm_data_gbm, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = blue) +
    geom_text(aes(label = Freq), vjust = 1) +
    labs(title = "GBM Confusion Matrix",
         x = "Actual Quality",
         y = "Predicted Quality",
         fill = "Frequency") +
    theme_light() +
    theme(plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "04d_GBM_ConfusionMatrix.png")
  
  ggsave(
    filename = Path,
    plot = plot_gbm_cm,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )
  
}, silent = TRUE)

}, silent = TRUE)

#==============================================================================#
#==== 04 - Model Comparison ===================================================#
#==============================================================================#

#==== 04a - Data ==============================================================#
## Collect the accuracy and RMSE scores per model used.

results_df <- data.frame(
  Model = c("Lasso", "Ridge", "Random Forest", "Gradient Boosting"),
  
  Accuracy = c(lasso_acc, 
               ridge_acc, 
               rf_final_acc, 
               gbm_final_acc),
  
  RMSE = c(lasso_test_rmse, 
           ridge_test_rmse, 
           rf_final_rmse, 
           gbm_final_rmse)
)

print(results_df)

#==== 04b - Visualisation =====================================================#

#=====================#
## Accuracy.
#=====================#

plot_acc <- ggplot(results_df, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", aes(fill = Model), width = 0.7) +
    geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5) +
  labs(title = "Final Model Accuracy",
       x = "Model",
       y = "Accuracy") +
  theme_light() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))

Path <- file.path(Charts_Directory, "05_Accuracy_Comparison.png")
ggsave(
  filename = Path,
  plot = plot_acc,
  width = 3750,
  height = 1833,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

#=====================#
## Combination.
#=====================#

# Path <- file.path(Charts_Directory, "07_Accuracy_Comparison.png")
# 
# ggsave(
#   filename = Path,
#   plot = plot,
#   width = 3750,
#   height = 1833,
#   units = "px",
#   dpi = 300,
#   limitsize = FALSE
# )

#==============================================================================#
#==== 05 - Appendix ===========================================================#
#==============================================================================#

#==== 05a - Implement the CatBoost Algorithm ==================================#

set.seed(123)

if(Enable_Catboost){
  
tryCatch({
  
#=======================#
## Parameters.
#=======================#

  train_x <- Train[, -which(names(Train) == "quality")]
  train_y <- Train$quality
  val_x   <- Validation[, -which(names(Validation) == "quality")]
  val_y   <- Validation$quality
  
  train_pool <- catboost.load_pool(data = train_x, label = train_y)
  val_pool   <- catboost.load_pool(data = val_x, label = val_y)
  
  Final_Train_Val <- rbind(Train, Validation) 
  
  final_train_x <- Final_Train_Val[, -which(names(Final_Train_Val) == "quality")]
  final_train_y <- Final_Train_Val$quality
  final_train_pool <- catboost.load_pool(data = final_train_x, label = final_train_y)
  
  test_x <- Test[, -which(names(Test) == "quality")]
  test_y <- Test$quality
  test_pool <- catboost.load_pool(data = test_x, label = test_y)
  
  param_grid <- expand.grid(
    learning_rate = c(0.03, 0.1),
    depth = c(10, 12, 14, 16),
    l2_leaf_reg = c(1, 5) 
  )
  
  tuning_results <- param_grid
  tuning_results$best_iteration <- NA
  tuning_results$validation_Accuracy <- NA
  
#=======================#
## Main Model.
#=======================#  
  
  print("--- STARTING HYPERPARAMETER TUNING (Classification) ---")
  for (i in 1:nrow(param_grid)) {
    
    params <- list(
      iterations = 2000,
      loss_function = 'MultiClass',
      eval_metric = 'Accuracy',
      train_dir = "my_run_logs",
      use_best_model = TRUE,
      early_stopping_rounds = 50,
      
      learning_rate = param_grid$learning_rate[i],
      depth = param_grid$depth[i],
      l2_leaf_reg = param_grid$l2_leaf_reg[i],
      logging_level = 'Silent'
    )
    
    model <- catboost.train(
      learn_pool = train_pool,
      test_pool = val_pool,
      params = params
    )
    
    ## read in.
    results_file <- file.path("my_run_logs", "test_error.tsv")
    run_metrics <- read.delim(results_file, sep = "\t")
    
    # 3. MAXIMIZE Accuracy
    best_idx <- which.max(run_metrics$Accuracy)
    best_accuracy <- run_metrics$Accuracy[best_idx]
    best_iter_value <- run_metrics$iter[best_idx] + 1
    
    tuning_results$validation_Accuracy[i] <- best_accuracy
    tuning_results$best_iteration[i] <- best_iter_value 
    
    print(paste("Completed run", i, "/", nrow(param_grid), 
                "| Best Iter:", tuning_results$best_iteration[i],
                "| Val. Accuracy:", round(tuning_results$validation_Accuracy[i], 5)))
  }
  
  print("--- TUNING COMPLETE ---")
  print(tuning_results)

#=======================#
## Final Model Assessment.
#=======================#  

  best_params_row <- tuning_results[which.max(tuning_results$validation_Accuracy), ]
  
  print("--- BEST HYPERPARAMETERS FOUND ---")
  print(best_params_row)
  
  final_params <- list(
    iterations = best_params_row$best_iteration, 
    loss_function = 'MultiClass',
    learning_rate = best_params_row$learning_rate,
    depth = best_params_row$depth,
    l2_leaf_reg = best_params_row$l2_leaf_reg,
    logging_level = 'Silent'
  )
  
  print("--- TRAINING FINAL MODEL ON ALL TRAIN/VAL DATA ---")
  final_model <- catboost.train(
    learn_pool = final_train_pool,
    params = final_params
  )
  
  test_preds <- catboost.predict(final_model, test_pool)
  print("--- FINAL MODEL EVALUATION ON TEST SET ---")
    test_rmse <- NA
    true_labels <- test_y #
  
  test_accuracy <- mean(test_preds == true_labels)
  print(paste("Test Set Accuracy:", round(test_accuracy, 5)))

#=======================#
## Visualization.
#=======================#    

  new_row <- data.frame(
    Model = "CatBoost",
    Accuracy = test_accuracy, 
    RMSE = test_rmse
  )
  
  results_df <- rbind(results_df, new_row)
  
  print("--- All Model Results ---")
  print(results_df)
  
  plot_acc <- ggplot(results_df, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
    geom_bar(stat = "identity", aes(fill = Model), width = 0.7) +
    geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5) +
    labs(title = "Final Model Accuracy", x = "Model", y = "Accuracy") +
    theme_light() +
    theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
  
  Path <- file.path(Charts_Directory, "05b_Accuracy_Comparison_with_CatBoost.png")
  ggsave(
    filename = Path,
    plot = plot_acc,
    width = 3750,
    height = 1833,
    units = "px",
    dpi = 300,
    limitsize = FALSE
  )

  tryCatch({
    
    importance_data_cb <- as.data.frame(catboost.get_feature_importance(final_model))
    importance_data_cb$Variable <- rownames(importance_data_cb)
    colnames(importance_data_cb)[1] <- "Importance"
    
    plot_cb_imp <- ggplot(importance_data_cb, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_bar(stat = "identity", fill = blue) +
      coord_flip() +
      labs(title = "Feature Importance (CatBoost Classification)",
           x = "Features",
           y = "Importance") +
      theme_light() +
      theme(plot.title = element_text(hjust = 0.5))
    
    Path <- file.path(Charts_Directory, "04e_CatBoost_FeatureImportance.png")
    
    ggsave(
      filename = Path,
      plot = plot_cb_imp,
      width = 3750,
      height = 1833,
      units = "px",
      dpi = 300,
      limitsize = FALSE
    )
    
  }, silent = TRUE)
  
  
  tryCatch({
    
    cm_data_cb <- as.data.frame(table(Actual = true_labels, Predicted = test_preds))
    
    plot_cb_cm <- ggplot(cm_data_cb, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white") +
      scale_fill_gradient(low = "white", high = blue) +
      geom_text(aes(label = Freq), vjust = 1) +
      labs(title = "CatBoost Confusion Matrix",
           x = "Actual Quality",
           y = "Predicted Quality",
           fill = "Frequency") +
      theme_light() +
      theme(plot.title = element_text(hjust = 0.5))
    
    Path <- file.path(Charts_Directory, "04f_CatBoost_ConfusionMatrix.png")
    
    ggsave(
      filename = Path,
      plot = plot_cb_cm,
      width = 3750,
      height = 1833,
      units = "px",
      dpi = 300,
      limitsize = FALSE
    )
    
  }, silent = TRUE)
  
}, silent = TRUE)

}

#==============================================================================#
#==============================================================================#
#==============================================================================#