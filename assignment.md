## Project Assignment

### Dataset
* Use the [US Accidents Dataset](https://canvas.wpi.edu/courses/87852/pages/project-knowledge-discovery-from-data) available at Kaggle.
* **Get very familiar with this dataset before you start working on the project.** Read the dataset page and the papers/documents linked from it.
* This dataset contains 7,728,394 data instances (accident records). To simplify your project a bit, use only the data instances that contain at most 1 missing value. According to our counts:
  * There are 3,554,549 data instances with no missing values.
  * There are 244,702 data instances with exactly one missing value.
  * Hence, your dataset for the project should contain 3,554,549 + 244,702 = 3,799,251 instances.
* **Python**: Run all project experiments in Python, using the Python packages and functions used in course activities and class.

---

### PART I: Data Exploration and Preprocessing
Work on this part in parallel with writing your project report. Put to good use what you learned from the Data Preprocessing Activity earlier this term.

#### Part I.1: Data Description and Exploration
* Provide a brief description of the dataset (e.g., dataset domain, number of instances, number of attributes, distribution of target attribute, attributes that have missing values, …).
* **Data Exploration:** Comments on interesting or salient aspects of the dataset, visualizations, correlation, issues with the data, …

#### Part I.2: Basic Data Preprocessing
* Determine which data pre-processing is strictly necessary before you can mine patterns from this dataset. For example, eliminate attributes that are identifiers. Keep this basic data pre-processing to a bare minimum.
* Apply this basic data preprocessing to the dataset before working on the other parts of this project.

#### Part I.3: Advanced Data Preprocessing
* Some of the data attributes contain very low level, granular data. Design ways to preprocess the data to make it more informative and to help uncover more meaningful and useful car accident patterns. Think like a domain expert: what data preprocessing will enable patterns that would be most helpful in explaining and/or preventing car accidents (e.g., combining attributes into richer attributes, aggregating values into higher-level concepts).
* Describe those ideas for data processing in your written report. The more insightful/clever your ideas, the better patterns you will obtain during mining.
* **Don't apply those more advanced data preprocessing ideas yet!** You will apply them within each of the other project parts to prevent data leakage and avoid affecting cross-validation.

---

### PART II: Classification
Work on this part in parallel with writing your project report. Put to good use what you learned from the Classification Activity and Evaluating Predictive Models Activity.

* **Classification Target:** Use `severity` as the target attribute.
* **Classification Techniques:**
  * **Majority Class Classifier:**
    * Use `DummyClassifier` from `sklearn.dummy` with default parameters and `strategy='most_frequent'` to predict the most frequent class in the training data.
  * **Decision Trees:**
    * Use `DecisionTreeClassifier` from `sklearn.tree`. Experiment with:
      * Default parameters, including `max_depth=None`.
      * `max_depth=5` and default values for the remaining parameters.
  * **Random Forests:**
    * Use `RandomForestClassifier` from `sklearn.ensemble`. Experiment with:
      * Default parameters, including `max_depth=None`.
      * `max_depth=5` and default values for the remaining parameters.
* **Evaluation Measures:** For each experiment, report:
  * Confusion matrix
  * Classification accuracy
  * Precision, recall, and f1-score for each class value (target attribute value)

#### Part II.1: Perform 10-fold Stratified Cross-Validation
* Use `model_selection.StratifiedKFold` to split the dataset into 10 stratified folds. Use these 10 folds for experimenting with Majority Class, Decision Trees, and Random Forests.
* Use `model_selection.cross_validate` to perform 10-fold cross-validation. At each of the 10 iterations:
  1. **IMPORTANT STEP:** Apply the advanced preprocessing designed in Part I.3 relevant for classification to this iteration's training set (9 folds) only.
  2. Build the prediction model for this iteration on the training set (9 folds) only.
  3. Apply the exact same advanced preprocessing applied to the training set to this iteration’s test set (the test fold).
  4. Evaluate this iteration’s model on the test set.
  5. Describe results in your report; include resulting evaluation measures for each iteration, along with the mean and standard deviation across iterations.

#### Part II.2: Final Classification Model
* Apply the advanced preprocessing designed in Part I.2 to the entire dataset.
* Construct a decision tree with `max_depth=5` and default values for remaining parameters on the entire dataset.
* What is the estimated classification accuracy of this decision tree over the entire domain of car accidents? Explain your answer.
* Include a depiction of the decision tree in your report.
* **Qualitative evaluation of the tree:**
  * How large is the tree?
  * How readable is the tree?
  * Describe the 3 most salient car accident patterns uncovered by this tree and how they can be useful to domain experts to understand and/or prevent car accidents.

---

### PART III: Regression
Work on this part in parallel with writing your project report. Put to good use what you learned from the Regression Activity and Evaluating Predictive Models Activity.

* **Regression Target:** Multiply the `severity` attribute values by 10, convert into a continuous (`float`) attribute, and use it as the target attribute.
* **Regression Techniques:**
  * **Majority Class Classifier:**
    * Use `DummyRegressor` from `sklearn.dummy` with default parameters and `strategy='mean'` to predict the average on the training data.
  * **Linear Regression:**
    * Use `LinearRegression` from `sklearn.linear_model` with default parameters.
  * **Regression Trees:**
    * Use `DecisionTreeRegressor` from `sklearn.tree`. Experiment with:
      * Default parameters, including `max_depth=None`.
      * `max_depth=5` and default values for remaining parameters.
  * **Random Forests:**
    * Use `RandomForestRegressor` from `sklearn.ensemble`. Experiment with:
      * Default parameters, including `max_depth=None`.
      * `max_depth=5` and default values for remaining parameters.
* **Evaluation Measures:** For each experiment, report Mean Squared Error (MSE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and $R^2$.

#### Part III.1: Perform 10-fold Cross-Validation
* Use `model_selection.KFold` to split the dataset into 10 folds. Use these 10 folds for experimenting with Majority Class, Decision Trees, and Random Forests.
* Use `model_selection.cross_validate` to perform 10-fold cross-validation. At each of the 10 iterations:
  1. **IMPORTANT STEP:** Apply the advanced preprocessing designed in Part I.3 relevant for regression to this iteration's training set (9 folds) only. *(Note: Impute missing values for Linear Regression only).*
  2. Build the prediction model for this iteration on the training set (9 folds) only.
  3. Apply the exact same advanced preprocessing applied to the training set to this iteration’s test set (the test fold).
  4. Evaluate this iteration’s model on the test set.
  5. Describe results in your report; include resulting evaluation measures for each iteration, along with the mean and standard deviation across iterations.

#### Part III.2: Final Regression Model
* Apply the advanced preprocessing designed in Part I.2 to the entire dataset.
* Construct a decision tree with `max_depth=5` and default values for remaining parameters on the entire dataset.
* What are the estimated generalization errors (MSE, RMSE, MAE, and $R^2$) of this regression tree over the entire domain of car accidents? Explain your answer.
* Include a depiction of the regression tree in your report.
* **Qualitative evaluation of the tree:**
  * How large is the tree?
  * How readable is the tree?
  * Describe the 3 most salient car accident patterns uncovered by this tree and how they can be useful to domain experts to understand and/or prevent car accidents.

---

### PART IV: Clustering
Work on this part in parallel with writing your project report.

* **Clustering Techniques:** Use the [scikit-learn clustering library](https://scikit-learn.org/stable/modules/clustering.html):
  * **K-means clustering:** `cluster.KMeans`
  * **Hierarchical clustering:** Experiment with `ward`, `complete`, `average`, and `single` linkage using `cluster.AgglomerativeClustering`.
  * **DBSCAN clustering:** Run experiments to find good parameter values using `cluster.DBSCAN`.
* **General Comments about Clustering:**
  * **Attribute Scaling:** Scale all continuous attributes to a 0 to 1 scale using `MinMaxScaler` so attributes with large ranges do not disproportionately affect distance/similarity measurements.
  * Vary algorithm parameters and provide in-depth evaluation and interpretation of results.
  * For K-means, plot SSE values for $k = 1, 2, 3, 4, 5, \dots$ to select a suitable $k$.
  * For DBSCAN, find good values for $\epsilon$ (`eps`) and `min_samples`.
  * Once the best $k$ for K-means is determined, compare the $k$-clustering obtained by K-means with the $k$-clusterings derived from hierarchical clustering dendrograms.
* **Clustering Evaluation:**
  * **Visual & Cluster Inspection:**
    * Visualization (e.g., MDS, t-SNE, UMAP, etc.) of resulting clusters.
    * Inspection of cluster members to find intra-cluster similarities and inter-cluster dissimilarities.
  * **Internal Indices:**
    * Sum of Squared Errors (SSE)
    * Heatmap of the correlation between the distance (or proximity) matrix and the incidence matrix
    * Silhouette Coefficient (`metrics.silhouette_score`)
  * **Relative Indices:**
    * Sum of Squared Errors (SSE)
    * Adjusted Rand score (`metrics.adjusted_rand_score`)
    * Normalized mutual information score (`metrics.normalized_mutual_info_score`)
    * Adjusted mutual information score (`metrics.adjusted_mutual_info_score`)
  * **External Indices:**
    * Compare clusterings with respect to the discrete target attribute `Severity` (values 1, 2, 3, 4). Ensure `Severity` is excluded from attributes used to construct clusters.
    * Homogeneity (`metrics.homogeneity_score`), Completeness (`metrics.completeness_score`), and V-measure (`metrics.v_measure_score`)
    * Contingency Matrix (`metrics.cluster.contingency_matrix`)

---

### PART V: Anomaly Detection using Cluster-based Methods
Work on this part in parallel with writing your project report.

* For each of the 3 clustering methods (K-means, hierarchical, and DBSCAN):
  * Select a clustering obtained from your experiments.
  * Define a sound anomaly score, $f(x)$, based on the clustering method.
  * Apply $f(x)$ to each data instance $x$ in the dataset.
  * Determine whether the selected clustering identified the presence of outliers (based on the clusters themselves or the anomaly score).
  * If outliers are identified, analyze what domain characteristics make them outliers.
  * Explain your findings, ideally illustrating with plots and/or visualizations.
  * Elaborate on whether the same outliers were identified across two or all three clustering methods.

---

### PART VI: Reflection
*This part can be completed individually or as a group.* Answer the following questions in your written report:

* **Reflect on the content (course material covered in this project):**
  * **(a)** What are the top 3 things you learned about data mining from working on this project?
  * **(b)** What are the top 3 things you learned about the data domain (car accidents) from this project?
  * **(c)** What are the top 3 things you want to learn more about regarding the data mining topics covered by this project?
* **Reflect on how you worked on this project:**
  * **(d)** If you could start working on this project again, what would you do differently?