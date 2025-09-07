Ensemble Model Based on Stacking: Demo 01
================
Donghui Xia (<dhxia@snut.edu.cn>)
2025-09-07

- [1 Introduction](#1-introduction)
- [2 Settings](#2-settings)
  - [2.1 Loading R packages](#21-loading-r-packages)
  - [2.2 Parallel Settings](#22-parallel-settings)
  - [2.3 Chinese Font Support](#23-chinese-font-support)
- [3 Data preparation](#3-data-preparation)
  - [3.1 Loading Raw Data](#31-loading-raw-data)
  - [3.2 Dataset splitting (stratified
    sampling)](#32-dataset-splitting-stratified-sampling)
- [4 Feature Selection](#4-feature-selection)
  - [4.1 Preprocessing Recipe (Initial recipe including all
    variables)](#41-preprocessing-recipe-initial-recipe-including-all-variables)
  - [4.2 Boruta Feature Selection](#42-boruta-feature-selection)
    - [4.2.1 Execution of Feature
      Selection](#421-execution-of-feature-selection)
    - [4.2.2 Extract Results](#422-extract-results)
    - [4.2.3 Visualizing](#423-visualizing)
    - [4.2.4 Exporting image](#424-exporting-image)
  - [4.3 Lasso Feature Selection](#43-lasso-feature-selection)
    - [4.3.1 Fitting LASSO Model](#431-fitting-lasso-model)
    - [4.3.2 Selection the best λ](#432-selection-the-best-λ)
    - [4.3.3 Plotting the cv lasso](#433-plotting-the-cv-lasso)
    - [4.3.4 Optimal lambda Values](#434-optimal-lambda-values)
    - [4.3.5 Building the final model using lambda.1se (more
      parsimonious
      model)](#435-building-the-final-model-using-lambda1se-more-parsimonious-model)
    - [4.3.6 Extracting important
      features](#436-extracting-important-features)
    - [4.3.7 Variable Importance
      Plotting](#437-variable-importance-plotting)
    - [4.3.8 Mapping one-hot encoded variables back to original
      variables](#438-mapping-one-hot-encoded-variables-back-to-original-variables)
  - [4.4 Recursive Feature Elimination
    (RFE)](#44-recursive-feature-elimination-rfe)
    - [4.4.1 Performing RFE for Feature
      Selection](#441-performing-rfe-for-feature-selection)
    - [4.4.2 3.4.2 RFE Features
      Plotting](#442-342-rfe-features-plotting)
  - [4.5 Random Forest for Feature
    Selection](#45-random-forest-for-feature-selection)
    - [4.5.1 Execute Random Forest (RF)](#451-execute-random-forest-rf)
    - [4.5.2 variable importance](#452-variable-importance)
    - [4.5.3 Plotting random forest variable
      importance](#453-plotting-random-forest-variable-importance)
  - [4.6 Feature Intersection](#46-feature-intersection)
    - [4.6.1 Obtain feature
      intersection](#461-obtain-feature-intersection)
    - [4.6.2 Plot Venn Diagram for Feature
      Sets](#462-plot-venn-diagram-for-feature-sets)
- [5 Model Data Preparation](#5-model-data-preparation)
  - [5.1 Modeling Dataset](#51-modeling-dataset)
  - [5.2 Modeling Preprocessing
    Recipes](#52-modeling-preprocessing-recipes)
    - [5.2.1 4.3创建交叉验证子集](#521-43创建交叉验证子集)
- [6 Conclusion](#6-conclusion)
- [7 Author](#7-author)

# 1 Introduction

This document demonstrates an ensemble model based on stacking
methodology.

# 2 Settings

## 2.1 Loading R packages

``` r
# Define required packages
required_packages <- c(
  "tidymodels",      # Modeling framework
  "tidyverse",       # Data manipulation and visualization
  "finetune",        # Model fine-tuning
  "stacks",          # Model stacking
  "caret",           # Classification and regression training
  "doFuture",        # Parallel execution with future
  "future",          # Parallel processing framework
  "future.apply",    # Parallel versions of base apply functions
  "furrr",           # Apply functions in parallel
  "Boruta",          # Feature selection
  "glmnet",          # Lasso and ridge regression
  "xgboost",         # Extreme gradient boosting
  "kernlab",         # Support vector machines (SVM)
  "kknn",            # Weighted k-nearest neighbors
  "ranger",          # Fast random forests
  "discrim",         # Gaussian naive Bayes
  "vip",             # Variable importance plots
  "ggvenn",          # Venn diagrams
  "pROC",            # ROC curves and AUC calculation
  "plotROC",         # ROC curve plotting
  "rmda",            # Decision curve analysis (DCA)
  "ggpubr",          # For arranging plots and publication-ready themes
  "ggsci",           # For professional color palettes
  "patchwork",       # Combining multiple plots
  "DALEXtra",        # Model explanation
  "shapviz",         # SHAP values for model interpretation
  "shiny",           # Interactive web applications
  "showtext",        # Chinese font support
  "tictoc",          # Code timing
  "ps"               # Process management
)

# Install missing packages without output
missing_packages <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE, quiet = TRUE)
}

# Load all packages silently
invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE, quietly = TRUE)
  )
}))

tidymodels_prefer()

# Loading custom functions
source("utils.R")
```

## 2.2 Parallel Settings

``` r
# Register doFuture to bind future with tune's parallel interface
registerDoFuture()

# Configure parallel execution strategy
plan(
  strategy = multisession,  # Use multisession strategy for parallel processing
  workers = max(1, availableCores() - 2)  # Reserve 2 cores for system operations
)

# Set future options for reproducible parallel processing
options(future.seed = TRUE)  # Ensure reproducibility in parallel computations
options(future.globals.maxSize = 0.8 * ps_system_memory()$total)  # Limit global variable size to 80% of total memory
```

## 2.3 Chinese Font Support

``` r
# Add Chinese font "SimSun" (宋体) for text rendering
font_add("MySong", "C:/Windows/Fonts/simsun.ttc") 

# Enable automatic text rendering with showtext
showtext_auto(enable = TRUE)
```

# 3 Data preparation

## 3.1 Loading Raw Data

``` r
df<- reading_data("attrition","modeldata") %>% 
  mutate(target = as.factor(Attrition)) %>%
  select(-Attrition)

table(df$target)
```

## 3.2 Dataset splitting (stratified sampling)

``` r
set.seed(723000)

split <- initial_split(df, strata = target, prop = 0.7)
train <- training(split)
test <- testing(split)
```

# 4 Feature Selection

## 4.1 Preprocessing Recipe (Initial recipe including all variables)

``` r
# Create base preprocessing recipe with all predictors
base_recipe <- recipe(target ~ ., data = train) %>%
  # Normalize all numeric predictors (center and scale)
  step_normalize(all_numeric_predictors()) %>%
  # Convert ordered factors to numeric scores
  step_ordinalscore(all_ordered_predictors()) %>%
  # Create dummy variables for all nominal predictors (excluding outcomes)
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  # Remove zero-variance predictors
  step_zv(all_predictors())
```

## 4.2 Boruta Feature Selection

### 4.2.1 Execution of Feature Selection

- The Boruta feature selection uses the original data.

``` r
# Set random seed to ensure reproducibility of results
set.seed(723000) 

tic("Boruta feature selection (with future parallel)")  
boruta_result <- Boruta(target ~ ., 
                        data = train,  
                        doTrace = 0,  # Set trace level to 0 (no process logging; 1 for basic logs, 2 for detailed logs)
                        pValue = 0.01)  # Set significance level to 0.01 (features with p-value ≤ 0.01 are considered significant)
toc()  

saveRDS(boruta_result, "fs_boruta_result.rds")  
```

### 4.2.2 Extract Results

``` r
# Load the saved Boruta feature selection results
boruta_result <- readRDS("fs_boruta_result.rds")  

# Extract and organize Boruta result data into a dataframe
# Extract and organize data - filter out shadow features
# 1. Identify shadow features (typically starting with "shadow")
shadow_cols <- grep("^shadow", colnames(boruta_result$ImpHistory), value = TRUE)

# 2. Keep only original (non-shadow) features
original_features <- setdiff(colnames(boruta_result$ImpHistory), shadow_cols)

# 3. Process feature importance data (only including original features)
imp_history <- boruta_result$ImpHistory[, original_features, drop = FALSE] %>%
  as.data.frame() %>%
  pivot_longer(everything(), names_to = "Feature", values_to = "Importance") %>%
  filter(is.finite(Importance))

# 4. Calculate median importance for each feature (for sorting)
median_imp <- imp_history %>%
  group_by(Feature) %>%
  summarise(Median_Importance = median(Importance), .groups = "drop") %>%
  arrange(Median_Importance) %>%
  mutate(Feature = factor(Feature, levels = Feature))

# 5. Add feature selection status (Confirmed/Tentative/Rejected)
feature_status <- data.frame(
  Feature = names(boruta_result$finalDecision),
  Status = as.character(boruta_result$finalDecision)
) %>%
  # Filter out status information for shadow features
  filter(Feature %in% original_features) %>%
  # Convert English status to Chinese
  mutate(
    Status = case_when(
      Status == "Confirmed" ~ "Confirmed",
      Status == "Tentative" ~ "Tentative",
      Status == "Rejected" ~ "Rejected",
      TRUE ~ Status  # Keep other possible statuses if any
    )
  )

# 6. Merge data
plot_data <- imp_history %>%
  left_join(feature_status, by = "Feature") %>%
  left_join(median_imp, by = "Feature") %>%
  mutate(Feature = factor(Feature, levels = median_imp$Feature))


boruta_vars <- getSelectedAttributes(boruta_result, withTentative = TRUE)  # Get selected features (including tentative ones)
cat("Number of features selected by Boruta:", length(boruta_vars),"\n")  # Print count of selected features
cat(boruta_vars,"\n")  # Print names of selected features

saveRDS(boruta_vars,"fs_boruta_vars.rds")  # Save selected feature names
```

### 4.2.3 Visualizing

``` r
# Create ggplot visualization with optimized color scheme
p <- ggplot(plot_data, aes(x = Feature, y = Importance)) +
  geom_boxplot(aes(fill = Status), width = 0.6, outlier.size = 0.5) +
  # Apply Nature journal color scheme from ggsci
  scale_fill_npg(
    name = "Feature Status",
    labels = c("Confirmed" = "Confirmed", "Tentative" = "Tentative", "Rejected" = "Rejected")
  ) +
  labs(
    x = "",
    y = "Feature Importance",
    fill = "Feature Status"
  ) +
  theme_pubr() +
  theme(
    text = element_text(family = "sans", size = 10.5),
    panel.border = element_rect(
      fill = NA,          # No fill inside panel border
      color = "black",    # Black border color
      linewidth = 0.5     # Border thickness
    ),
    axis.ticks.y = element_line(
      color = "black",    # Y-axis tick color
      linewidth = 0.6     # Y-axis tick thickness
    ),
    axis.text.y = element_text(
      color = "black",    # Y-axis text color
      size = 9            # Y-axis text size
    ),
    axis.ticks.x = element_line(
      color = "black",    # X-axis tick color
      linewidth = 0.6     # X-axis tick thickness
    ),
    axis.text.x = element_text(
      angle = 60,         # Rotate X-axis labels by 45 degrees
      hjust = 1,          # Horizontal alignment
      vjust = 1,          # Vertical alignment
      size = 9            # X-axis text size
    ),
    legend.position = c(0.05, 0.95),  # Legend position (top-left)
    legend.justification = c(0, 1),   # Legend justification anchor
    legend.background = element_rect(
      fill = "white",    # White background for legend
      color = NA         # No border around legend
    ),
    legend.direction = "horizontal",  # Arrange legend horizontally
    legend.key.size = unit(0.6, "cm"),  # Size of legend keys
    legend.text = element_text(size = 9),  # Legend text size
    legend.title = element_text(size = 9),  # Legend title size
    legend.spacing.x = unit(0.8, "cm")  # Horizontal spacing between legend items
  )

print(p)
```

### 4.2.4 Exporting image

``` r
# Save the plot as an image with dimensions 13.5cm × 9cm
ggsave(
  # "boruta_result_300dpi2.tiff",  # Optional: TIFF format (commented out)
  "boruta_result_300dpi.svg",     # Output file name (SVG format, vector graphics)
  plot = p,                       # The ggplot object to save (created earlier)
  # device = cairo_pdf,            # Optional: Use Cairo PDF device for better font rendering (commented out)
  # device = agg_tiff,             # Optional: Use agg_tiff device for high-quality TIFF output (commented out)
  width = 13.5,                   # Image width: 13.5 cm (matches typical Word page width)
  height = 9,                     # Image height: 9 cm
  units = "cm",                   # Unit of measurement for width/height
  # compression = "lzw",           # Optional: Use LZW lossless compression (for TIFF; commented out)
  dpi = 300                       # Resolution: 300 DPI (meets publication quality standards)
)
```

## 4.3 Lasso Feature Selection

<https://mp.weixin.qq.com/s/GhpYRGNckgJUwshZ9lRbTg>

### 4.3.1 Fitting LASSO Model

``` r
lasso_recipe <- prep(base_recipe, training = train)
lasso_train <- bake(lasso_recipe, new_data = train)
x <- as.matrix(lasso_train[, -which(names(lasso_train) == "target")])
y <- lasso_train$target

tic("Lasso Feature Selection")
set.seed(723000)
cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial", nfolds = 10)
toc()

saveRDS(cv_lasso,"fs_cv_lasso.rds")
```

### 4.3.2 Selection the best λ

``` r
cv_lasso <- readRDS("fs_cv_lasso.rds")
best_lambda <- cv_lasso$lambda.1se
best_lambda
```

### 4.3.3 Plotting the cv lasso

``` r
# Original plotting function: plot(cv_lasso)
# Prepare cross-validation data for visualization
cv_data <- data.frame(
  lambda = cv_lasso$lambda,                # Lambda values used in cross-validation
  mean_loss = cv_lasso$cvm,                # Mean cross-validation error (deviance)
  se = cv_lasso$cvsd,                      # Standard error of the cross-validation error
  upper = cv_lasso$cvm + cv_lasso$cvsd,    # Upper bound of error (mean + SE)
  lower = cv_lasso$cvm - cv_lasso$cvsd     # Lower bound of error (mean - SE)
) %>%
  arrange(desc(lambda))  # Sort by descending lambda values

# Calculate maximum y-value for annotation positioning
y_max <- max(cv_data$upper)
neg_log_lambda <- -log(lambda)  # Negative log transformation of lambda

# Filter out data points where number of features is zero
filtered_data <- axis_data[axis_data$num_features != 0, ]

# Function to find the closest index to a target value
find_closest_index <- function(target, values) {
  which.min(abs(values - target))  # Return index of value closest to target
}

n_breaks <- 6  # Number of breaks for x-axis

# Generate evenly distributed target positions for x-axis breaks
x_breaks_target <- seq(min(filtered_data$neg_log_lambda), 
                      max(filtered_data$neg_log_lambda), 
                      length.out = n_breaks)

# Find actual data points closest to target break positions
closest_indices <- sapply(x_breaks_target, find_closest_index, values = filtered_data$neg_log_lambda)
closest_neg_log_lambda <- filtered_data$neg_log_lambda[closest_indices]
closest_num_features <- filtered_data$num_features[closest_indices]

# Define consistent theme settings for both plots
uniform_theme <- theme(
  text = element_text(family = "sans"),
  axis.text = element_text(size = 8),          # Axis tick label size
  axis.title = element_text(size = 10),        # Axis title size
  panel.grid = element_blank(),                # Remove grid lines
  panel.border = element_rect(color = "black", linewidth = 0.5, fill = NA)  # Add border
)

# Create first plot (cross-validation error vs lambda)
p2 <- ggplot(cv_data, aes(x = neg_log_lambda, y = mean_loss)) +
    geom_point(color = pal_aaas()(5)[2], size = 1, shape = 16) +  # Plot mean error points
    geom_line(color = pal_aaas()(5)[2], linewidth = 0.25) +       # Connect points with line
    geom_errorbar(                                                # Add error bars (mean ± SE)
        aes(ymin = lower, ymax = upper),
        color = pal_uchicago()(9)[8],
        width = 0.05,
        linewidth = 0.5
    ) +
    geom_vline(                                                   # Vertical line for lambda.min
        xintercept = -log(cv_lasso$lambda.min),
        linetype = "dashed",
        color = pal_uchicago()(9)[8],
        linewidth = 0.5
    ) + 
    annotate(                                                     # Label for lambda.min
      "text",
      x = -log(cv_lasso$lambda.min),
      y = y_max * 0.95,  # Position near top of plot
      label = expression(lambda[min]),
      color = pal_aaas()(5)[2],
      hjust = 1.1,       # Adjust horizontal justification
      vjust = 1,
      size = 3.5         # Text size
    ) +
    geom_vline(                                                   # Vertical line for lambda.1se
        xintercept = -log(cv_lasso$lambda.1se),
        linetype = "dashed",
        color = pal_uchicago()(9)[8],
        linewidth = 0.5
    ) +
    annotate(                                                     # Label for lambda.1se
      "text",
      x = -log(cv_lasso$lambda.1se),
      y = y_max * 0.95,  # Position near top of plot
      label = expression(lambda["1se"]),
      color = pal_aaas()(5)[2],
      hjust = -0.1,      # Adjust horizontal justification
      vjust = 1,
      size = 3.5 
    ) +
    scale_x_continuous(
        name = expression(-log(lambda)),  # X-axis label
        trans = "reverse",                # Reverse x-axis (lambda decreases left to right)
        breaks = scales::pretty_breaks(n = 6)  # Automatic pretty breaks
    ) +
    labs(
        y = "Binomial Deviance"  # Y-axis label (loss function for binary outcomes)
    ) +
    theme_pubr(base_family = "sans") +  # Publication-ready theme
    uniform_theme                       # Apply consistent theme

# Create second plot (coefficient paths vs lambda)
p3 <- ggplot(coef_data, aes(x = neg_log_lambda, y = Coefficient, 
                                group = Feature, color = Feature)) +
  geom_line(linewidth = 0.5) +  # Plot coefficient paths for each feature
  scale_color_manual(values = my_colors) +  # Use custom color palette
  
  scale_x_continuous(
    name = expression(-log(lambda)),        # Primary x-axis label
    trans = "reverse",                      # Reverse x-axis
    breaks = scales::pretty_breaks(n_breaks),  # Consistent breaks with first plot
    sec.axis = sec_axis(                    # Secondary x-axis showing number of features
      trans = ~.,
      name = "Number of Features",
      breaks = closest_neg_log_lambda,      # Match breaks to primary axis
      labels = closest_num_features         # Show corresponding feature counts
    )
  ) +
  
  scale_y_continuous(
    name = "Coefficient",                   # Y-axis label
    breaks = seq(                           # Custom breaks for coefficient values
      floor(min(coef_data$Coefficient)),
      ceiling(max(coef_data$Coefficient)),
      by = 0.5
    )
  ) +
  
  # Add vertical reference lines for optimal lambdas
  geom_vline(
    xintercept = neg_log_lambda_min,
    color = pal_uchicago()(9)[8],
    linetype = "dashed",
    linewidth = 0.5
  ) +
  geom_vline(
    xintercept = neg_log_lambda_1se,
    color = pal_uchicago()(9)[8],
    linetype = "dashed",
    linewidth = 0.5
  ) +
  
  # Add annotations for lambda labels
  annotate(
    "text",
    x = neg_log_lambda_min,
    y = max(coef_data$Coefficient) * 0.95,  # Position near top
    label = expression(lambda[min]),
    color = pal_aaas()(5)[2],
    hjust = 1.1,
    vjust = 1,
    size = 3.5
  ) +
  annotate(
    "text",
    x = neg_log_lambda_1se,
    y = max(coef_data$Coefficient) * 0.95,  # Position near top
    label = expression(lambda["1se"]),
    color = pal_aaas()(5)[2],
    hjust = -0.1,
    vjust = 1,
    size = 3.5
  ) +
  
  guides(color = "none") +  # Remove color legend
  theme_pubr() +            # Publication-ready theme
  uniform_theme +           # Apply consistent theme
  theme(
    # Additional theme adjustments specific to second plot
    axis.line.x.bottom = element_line(color = "black", linewidth = 0.5),
    axis.line.y.right = element_line(color = "black", linewidth = 0.5),
    axis.text.x.top = element_text(margin = margin(b = 5), size = 8),
    axis.title.x.top = element_text(size = 8),
    axis.ticks.x.top = element_line()
  )

# Ensure consistent margins between plots
p2 <- p2 + theme(plot.margin = margin(t = 10, r = 5, b = 5, l = 20, unit = "pt"))
p3 <- p3 + theme(plot.margin = margin(t = 10, r = 5, b = 5, l = 20, unit = "pt"))

# Combine both plots vertically
combined_plot <- ggarrange(
  p2, p3,
  labels = c("A", "B"),       # Panel labels
  ncol = 1, nrow = 2,         # Stack vertically (1 column, 2 rows)
  align = "v",                # Align vertically
  common.legend = FALSE,      # No shared legend
  legend = "bottom",
  heights = c(1, 1.2),        # Relative heights of plots
  label.x = 0.02,             # Adjust label position
  label.y = 0.65,
  hjust = 0,
  vjust = 1,
  font.label = list(size = 12, family = "sans")  # Consistent label font
)

print(combined_plot)  # Display the combined plot

# Save the plot as an SVG file
ggsave(
  "combined_analysis_plot.svg",
  combined_plot,
  width = 13.5, height = 9, units = "cm",  # Dimensions
  dpi = 300                                # Resolution
)
```

### 4.3.4 Optimal lambda Values

``` r
cat("Optimal lambda (lambda.min):", cv_lasso$lambda.min,"\n")
cat("Lambda within 1 standard error (lambda.1se):", cv_lasso$lambda.1se,"\n")
```

### 4.3.5 Building the final model using lambda.1se (more parsimonious model)

``` r
# Fit the final LASSO model using the lambda.1se value
set.seed(723000)

lasso_model <- glmnet(
  x, 
  y,
  alpha =1,
  family ="binomial",
  lambda = cv_lasso$lambda.1se
)
```

### 4.3.6 Extracting important features

``` r
# Extract coefficients corresponding to the best lambda
lasso_coef <- coef(lasso_model)

# Identify variables with non-negligible coefficients (excluding intercept)
# Using 1e-8 as a threshold to account for floating point precision
lasso_vars <- rownames(lasso_coef)[which(abs(lasso_coef) >= 1e-8)]
lasso_vars <- lasso_vars[lasso_vars != "(Intercept)"]
saveRDS(lasso_vars, "fs_lasso_vars.rds")  # Save selected features

# Create data frame with features and their coefficients
lasso_features <- data.frame(  
  Feature = rownames(lasso_coef)[which(abs(lasso_coef) >= 1e-8)],  
  Coefficient = as.vector(lasso_coef[which(abs(lasso_coef) >= 1e-8)])  # Convert matrix to vector
)
lasso_features  # Display the features and their coefficients
```

### 4.3.7 Variable Importance Plotting

``` r
# Create a plot to visualize feature importance from the LASSO model

p4 <- ggplot(
  lasso_features, 
  aes(x = reorder(Feature, Coefficient), y = Coefficient, 
      fill = ifelse(Coefficient > 0, "Positive", "Negative"))
) +  
  geom_col(width = 0.7, alpha = 0.9) +
  geom_text(
    aes(label = sprintf("%.3f", Coefficient), 
        hjust = ifelse(Coefficient > 0, -0.2, 1.2)),
    size = 2,
    color = "black"
  ) +
  scale_fill_jama(
    # name = "Effect Direction",
    name = NULL,
    labels = c("Negative", "Positive")
  ) +
  scale_y_continuous(
    expand = expansion(mult = 0.1),  
    limits = c(
      min(lasso_features$Coefficient) * 1.1, 
      max(lasso_features$Coefficient) * 1.1   
    ),
    name = "Coefficient"
  ) +
  coord_flip() +
  labs(
    # title = "LASSO Feature Importance",
    x = "Features"
  ) +
  theme_pubr(base_family = "sans") +
  theme(
    plot.title = element_text(size = 5, 
                              # face = "bold", 
                              hjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 7),
    legend.position = c(0.7,0.4),
    legend.title = element_text(face = "bold", size = 8),  # 图例标题大小
    legend.text = element_text(size = 7)  # 图例标签大小
  ) +
  guides(fill = guide_legend(
    keywidth = unit(0.3, "cm"),   # 图例键宽度
    keyheight = unit(0.3, "cm")   # 图例键高度
  )
  )

print(p4)

# Save the plot as an SVG file
ggsave(
  "LASSO Feature Importance.svg",
  p4,
  width = 13.5, height = 9, units = "cm",  # Dimensions
  dpi = 300                                # Resolution
)
```

### 4.3.8 Mapping one-hot encoded variables back to original variables

``` r
# Get dummy variable information from the recipe
dummy_info <- tidy(lasso_recipe, number = which(sapply(lasso_recipe$steps, function(x) class(x)[1] == "step_dummy")))

# Create mapping function: convert dummy variables to original variables
map_dummy_to_original <- function(dummy_vars, dummy_info) {
  original_vars <- c()
  
  for (var in dummy_vars) {
    # Check if it's a numeric variable (not one-hot encoded)
    if (!grepl("_", var)) {
      original_vars <- c(original_vars, var)
      next
    }
    
    # Process one-hot encoded variables
    found <- FALSE
    for (i in 1:nrow(dummy_info)) {
      pattern <- paste0("^", dummy_info$terms[i], "_")
      if (grepl(pattern, var)) {
        original_vars <- c(original_vars, dummy_info$terms[i])
        found <- TRUE
        break
      }
    }
    
    if (!found) {
      original_vars <- c(original_vars, var)
    }
  }
  
  unique(original_vars)  # Remove duplicates
}

# Apply mapping function
lasso_original_vars <- map_dummy_to_original(lasso_vars, dummy_info)


# Display results
cat("Number of features selected by Lasso：", length(lasso_original_vars), "\n")
cat(lasso_original_vars, "\n")

# Save original variables
saveRDS(lasso_original_vars,"fs_lasso_original_vars.rds")
```

## 4.4 Recursive Feature Elimination (RFE)

### 4.4.1 Performing RFE for Feature Selection

``` r
# Set RFE control parameters
ctrl <- rfeControl(
  functions = rfFuncs,  # Use random forest as the base model for feature evaluation
  method = "cv",        # Use 10-fold cross-validation to assess feature subsets
  number = 10 
  )

# Execute RFE
set.seed(723000)  # Ensure reproducibility

tic("RFE Feature Selection")  
rfe_result <- rfe(
  x = train %>% select(-target), 
  y = train$target,               
  sizes = seq(1,ncol(x),1),        
  rfeControl = ctrl               
)
toc() 

# Save RFE results
saveRDS(rfe_result,"fs_rfe.rds")
```

### 4.4.2 3.4.2 RFE Features Plotting

``` r
rfe_result <- readRDS("fs_rfe.rds")

rfe_vars <- predictors(rfe_result)
rfe_vars
saveRDS(rfe_vars,"fs_rfe_vars.rds")


# Plot RFE results (shows performance vs feature subset size)
# plot(rfe_result, type = c("g", "o"))

# 提取RFE结果数据
rfe_data <- data.frame(
  Variables = rfe_result$results$Variables,
  Accuracy = rfe_result$results$Accuracy,
  Kappa = rfe_result$results$Kappa
)

# 找到最佳特征数
best_index <- which.max(rfe_data$Accuracy)
best_variables <- rfe_data$Variables[best_index]
best_accuracy <- rfe_data$Accuracy[best_index]

# 创建ggplot2图形
p5 <- ggplot(rfe_data, aes(x = Variables, y = Accuracy)) +
  geom_line(color = pal_aaas()(1), linewidth = 0.5) +
  geom_point(color = pal_aaas()(1), size = 1, shape = 19) +
  geom_point(data = rfe_data[best_index, ], 
             aes(x = Variables, y = Accuracy),
             color = pal_lancet()(1), 
             size = 1.5, 
             shape = 16, 
             stroke = 1.5,
             fill = "white") +
  geom_vline(xintercept = best_variables, 
             linetype = "dashed", 
             color = pal_lancet()(1),
             linewidth = 0.5) +
  geom_hline(yintercept = best_accuracy, 
             linetype = "dashed", 
             color = pal_lancet()(1),
             linewidth = 0.5) +
  annotate("text", 
           x = best_variables, 
           y = min(rfe_data$Accuracy),
           label = paste("Optimal:", best_variables, "features"),
           color = pal_lancet()(1),
           vjust = -1,
           fontface = "bold",
           size = 4) +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    # title = "Recursive Feature Elimination (RFE) Results",
    # subtitle = "Feature subset selection using Random Forest",
    x = "Number of Features",
    y = "Cross-Validation Accuracy"
  ) +
  theme_pubr(base_size = 10, base_family = "sans") +
  theme(
    # plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    # plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 12),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    panel.grid = element_blank()
  )

print(p5)

ggsave(
  "Recursive Feature Elimination (RFE) Results.svg",
  p5,
  width = 13.5, height = 9, units = "cm",  # Dimensions
  dpi = 300                                # Resolution
)
```

## 4.5 Random Forest for Feature Selection

### 4.5.1 Execute Random Forest (RF)

``` r
# Set seed for reproducibility of results
set.seed(723000)

# Start timing the random forest feature selection process
tic("Random Forest Feature Selection")

# Create and train a random forest model
rf_model <- rand_forest(
  mode = "classification",  
  trees = 1000) %>%      
  set_engine("ranger", importance = "permutation") %>%  
  fit(target ~ ., data = train) 
toc()

saveRDS(rf_model,"fs_rf_model.rds")
```

### 4.5.2 variable importance

``` r
rf_model <- readRDS("fs_rf_model.rds")

rf_imp <- vip::vi(rf_model)
rf_vars <- rf_imp %>%
  filter(Importance > quantile(Importance, 0.5)) %>% 
  pull(Variable)

saveRDS(rf_vars,"fs_rf_vars.rds")
```

### 4.5.3 Plotting random forest variable importance

``` r
imp_data <- vip::vi(rf_model) %>% filter(Importance > quantile(Importance, 0.5))

p6 <- ggplot(imp_data, 
             aes(x = Importance, 
                 y = reorder(Variable, Importance), 
                 size = Importance, 
                 color = Importance)) +
    geom_point(alpha = 0.7) +
    
    scale_color_gradientn(colors = pal_jama()(7)) +

    scale_size_continuous(range = c(3, 10), guide = "none") + 
    
    labs(x = "Importance Score",
         y = "Variables",
         color = "Importance") + 
    scale_y_discrete(expand = expansion(mult = c(0.05, 0.15))) +
    theme_pubr(base_family = "sans",base_size = 10) +
    theme(
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
        axis.text.y = element_text(size = 8),
        axis.text.x = element_text(size = 8),
        legend.position = c(0.85, 0.4), 
        legend.background = element_blank(),  
        legend.key = element_blank(),         
        panel.background = element_blank(),   
        plot.background = element_blank(),
        legend.title = element_text(size = 8), 
        legend.text = element_text(size = 7),
        plot.margin = margin(10, 10, 10, 10) 
    ) +
    
    guides(
        color = guide_colorbar(
          title.position = "top",
          title.hjust = 0.5,
          barwidth = unit(0.5, "cm"),
          barheight = unit(5, "cm"),
          frame.colour = "black", 
          ticks.colour = "black")  
    ) +
    
    annotate("text", 
             x = max(imp_data$Importance) * 0.4,
             y = 0.75,
             label = paste("Total variables:", nrow(imp_data), 
                           "\nMax importance:", round(max(imp_data$Importance), 3),
                           "\nMin importance:", round(min(imp_data$Importance), 3)),
             hjust = 0, vjust = 0,
             size = 2.5,
             color = "gray30",
             fontface = "italic") 

print(p6)

ggsave(
  "random forest variable importance.svg",
  p6,
  width = 13.5, height = 9, units = "cm",  # Dimensions
  dpi = 300                                # Resolution
)
```

## 4.6 Feature Intersection

### 4.6.1 Obtain feature intersection

``` r
boruta_vars <- readRDS("fs_boruta_vars.rds")
lasso_original_vars <- readRDS("fs_lasso_original_vars.rds")
rfe_vars <- readRDS("fs_rfe_vars.rds")
rf_vars <- readRDS("fs_rf_vars.rds")


feature_list <- list(
  Boruta = boruta_vars,
  Lasso = lasso_original_vars, 
  RFE = rfe_vars, 
  RF = rf_vars
)

feature_list <- feature_list[sapply(feature_list, length) > 0]


selected_vars <- reduce(feature_list, intersect)

selected_vars

saveRDS(selected_vars,"selected_vars.rds")
```

### 4.6.2 Plot Venn Diagram for Feature Sets

``` r
p7 <-  ggvenn(feature_list, 
       fill_color = c("blue", "yellow", "green", "red"),
       stroke_size = 1,
       set_name_size = 0,  # 先隐藏原有标签
       text_size = 5,
       show_outside = "none",
       show_percentage = FALSE) +
  # 手动添加集合标签
  annotate("text", x = -1, y = 1.15, label = names(feature_list)[1], 
           size = 5, family = "sans", color = "black") +
  annotate("text", x = 1, y = 1.15, label = names(feature_list)[2], 
           size = 5, family = "sans", color = "black") +
  annotate("text", x = -1.6, y = 0.8, label = names(feature_list)[3], 
           size = 5, family = "sans", color = "black") +
  annotate("text", x = 1.6, y = 0.8, label = names(feature_list)[4], 
           size = 5, family = "sans", color = "black") +
  theme(text = element_text(family = "sans", size = 10))

print(p7)

ggsave(
  "venn_plot_300dpi.svg",
  plot = p7,
  width = 13.5,   
  height = 9,     
  units = "cm",
  dpi = 300
)
```

# 5 Model Data Preparation

## 5.1 Modeling Dataset

``` r
# Load the pre-selected features from previous feature selection step
selected_vars <- readRDS("selected_vars.rds")

# Create training dataset with only selected features and target variable
train_fs <- train[c(selected_vars, "target")]

# Create test dataset with the same selected features and target variable
test_fs <- test[c(selected_vars, "target")]
```

## 5.2 Modeling Preprocessing Recipes

| Model | Preprocessing Recipe | Standardization | Dummy Encoding | Ordinal Factor Handling | Nominal Factor Handling |
|:---|:---|:---|:---|:---|:---|
| Logistic Regression | dummy_recipe | Yes | Yes | step_ordinalscore | step_dummy |
| KNN | dummy_recipe | Yes | Yes | step_ordinalscore step_dummy |  |
| Linear SVM | dummy_recipe | Yes | Yes | step_ordinalscore | step_dummy |
| RBF SVM | dummy_recipe | Yes | Yes | step_ordinalscore | step_dummy |
| Random Forest | tree_recipe | No | No | step_ordinalscore | Keep as factor |
| XGBoost | tree_recipe | No | No | step_ordinalscore | Keep as factor |
| LightGBM | tree_recipe | No | No | step_ordinalscore | Keep as factor |
| Neural Network | dummy_recipe | Yes | Yes | step_ordinalscore | step_dummy |

Note: For XGBoost, LightGBM, and Neural Network (MLP) models, target
variable conversion from factor to numeric will be handled in their
respective workflow setups as follows: add_step(step_mutate(target =
as.numeric(target) - 1)) \# For binary classification (0/1)

``` r
# Load required packages
library(recipes)

# 1. Dummy recipe for models requiring standardization and dummy encoding
dummy_recipe <- recipe(target ~ ., data = train_fs) %>%
  # Convert ordinal factors to numerical scores
  step_ordinalscore(all_nominal_predictors(), -all_outcomes()) %>%
  # Create dummy variables for nominal factors
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  # Standardize numerical predictors (mean = 0, SD = 1)
  step_normalize(all_numeric_predictors(), -all_outcomes())

# 2. Tree-based recipe for models not requiring standardization/dummy encoding
tree_recipe <- recipe(target ~ ., data = train_fs) %>%
  # Convert ordinal factors to numerical scores
  step_ordinalscore(all_nominal_predictors(), -all_outcomes())
  # Nominal factors remain as factors for tree-based models
  # No standardization for tree-based models
```

### 5.2.1 4.3创建交叉验证子集

``` r
# 创建交叉验证折数
set.seed(2025723001)
folds <- vfold_cv(train_fs, v = 10, strata = target)  # 变更数据集和目标变量、折数

cat("创建了", length(folds$splits), "折交叉验证子集\n")
```

# 6 Conclusion

This utility function provides a convenient way to read multiple data
formats with a consistent interface. It’s particularly useful for:

- Data analysis workflows
- Teaching materials with diverse data sources
- Projects requiring data from multiple formats

For questions or suggestions about this function, please contact the
author:

# 7 Author

Donghui Xia<br> Email: <dhxia@snut.edu.cn>  
ORCID: [0000-0002-2264-7596](https://orcid.org/0000-0002-2264-7596)<br>
[School of Chemical and Environmental Science](), [Shaanxi University of
Technology]()<br> [Shaanxi key Laboratory of Catalysis]()
