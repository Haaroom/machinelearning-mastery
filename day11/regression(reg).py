from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
data = fetch_california_housing()
X = data.data.astype("float32")
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_Scaled = sc.fit_transform(X_train)
X_test_Scaled = sc.transform(X_test)
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred)
    }
results_list = []
for name, model in models.items():  
    model.fit(X_train_Scaled, y_train)
    y_pred = model.predict(X_test_Scaled)
    metrics = evaluate(y_test, y_pred)
    results_list.append({
        "Model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"]
    })
df = pd.DataFrame(results_list)
print(df)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)
fig, axes = plt.subplots(1, 2)
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(
    data=df_melted, 
    x="Metric", 
    y="Score", 
    hue="Model", 
    ax=axes[0], 
    palette="muted"
)
axes[0].set_title("Model Performance Comparison")
axes[0].set_ylabel("Metric Value")
axes[0].set_xlabel("")
best_model = models["Linear"]
y_pred_best = best_model.predict(X_test_Scaled)
sns.scatterplot(
    x=y_test, 
    y=y_pred_best, 
    alpha=0.3, 
    color="purple", 
    ax=axes[1]
)
axes[1].plot(
    [y_test.min(), y_test.max()], 
    [y_test.min(), y_test.max()], 
    color="red", 
    linestyle="--", 
    linewidth=2
)
axes[1].set_title("Linear Regression: Actual vs. Predicted Prices")
axes[1].set_xlabel("Actual House Value ($100ks)")
axes[1].set_ylabel("Predicted House Value ($100ks)")
plt.tight_layout()
plt.show()
