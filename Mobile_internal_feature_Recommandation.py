# # 📱Mobile Phone Internal  Features Recommendation System
# # Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.ticker as mtick

sns.set(style='whitegrid')

# # Step 2: Load Dataset
# # url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Mobile%20Price%20Classification.csv'
# df = pd.read_csv(url)

df = pd.read_csv('mobile_price_data.csv')

# Show the first few rows to confirm
df.head()

# # Step 3: Data Cleaning
print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.rename(columns={'price_range': 'label'}, inplace=True)

# # Step 4: Exploratory Data Analysis (EDA)

# # 4.1 Countplot
plt.figure(figsize=(8,5))
sns.countplot(x='label', data=df, palette='pastel')
plt.title("Price Range Distribution")
plt.xlabel("Price Range (0: Low, 3: High)")
plt.ylabel("Count")
plt.show()

# # 4.2 Pie Chart
plt.figure(figsize=(7,7))
labels = ['Low Cost (0)', 'Medium Cost (1)', 'High Cost (2)', 'Very High Cost (3)']
sizes = df['label'].value_counts().sort_index()
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
plt.title("Mobile Price Range Distribution")
plt.axis('equal')
plt.show()

# # 4.3 Boxplot RAM vs Price
plt.figure(figsize=(8,5))
sns.boxplot(x='label', y='ram', data=df, palette='coolwarm')
plt.title("RAM by Price Range")
plt.xlabel("Price Range")
plt.ylabel("RAM")
plt.show()

# # 4.4 Correlation Heatmap
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# # 4.5 RAM vs Battery Power
plt.figure(figsize=(8,5))
sns.scatterplot(x='ram', y='battery_power', hue='label', data=df, palette='Set2')
plt.title("RAM vs Battery Power by Price Range")
plt.xlabel("RAM")
plt.ylabel("Battery Power")
plt.show()

# # 4.6 Feature Histograms
features_to_plot = ['ram', 'battery_power', 'px_height', 'px_width']
for feature in features_to_plot:
    plt.figure(figsize=(8,4))
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f"{feature} Distribution")
    plt.show()

# # 4.7 Pairplot
sample_df = df[['ram', 'battery_power', 'px_width', 'px_height', 'label']]
sns.pairplot(sample_df, hue='label', palette='Set1')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# # Step 5: Feature and Target Split
X = df.drop(columns='label')
y = df['label']

# # Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 7: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Step 8: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# # Step 9: Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 9.1 Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9.2 Classification Report Visualization
report = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1,2,3])
metrics_df = pd.DataFrame({
    'Class': [0, 1, 2, 3],
    'Precision': report[0],
    'Recall': report[1],
    'F1-Score': report[2]
})
metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(8,5), colormap='Set2')
plt.title("Classification Report - Per Class")
plt.ylim(0, 1.1)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# # Step 10: Feature Importance
plt.figure(figsize=(10,6))
pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show()

# # Step 11: Prediction Simulation
sample_input = {
    'battery_power': 1200, 'blue': 1, 'clock_speed': 2.2, 'dual_sim': 1, 'fc': 5,
    'four_g': 1, 'int_memory': 32, 'm_dep': 0.6, 'mobile_wt': 160, 'n_cores': 4,
    'pc': 13, 'px_height': 600, 'px_width': 1000, 'ram': 4000, 'sc_h': 10, 'sc_w': 6,
    'talk_time': 15, 'three_g': 1, 'touch_screen': 1, 'wifi': 1
}
sample_df = pd.DataFrame([sample_input])
sample_scaled = scaler.transform(sample_df)
predicted_range = model.predict(sample_scaled)[0]
print(f"\n📱 Predicted Price Range Class: {predicted_range} (0=Low, 3=High)")

# # Step 12: User-Based Collaborative Filtering
ratings_data = {
    'user': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
    'mobile_id': [1, 2, 3, 2, 3, 1, 4, 2, 5, 4, 5],
    'rating': [5, 4, 3, 5, 4, 4, 5, 4, 3, 4, 5]
}
ratings_df = pd.DataFrame(ratings_data)
user_item_matrix = ratings_df.pivot_table(index='user', columns='mobile_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_mobiles(user, user_item_matrix, n_recommendations=3):
    similar_users = user_similarity_df[user].sort_values(ascending=False).drop(user)
    top_user = similar_users.index[0]
    target_ratings = user_item_matrix.loc[user]
    top_user_ratings = user_item_matrix.loc[top_user]
    recommendations = top_user_ratings[target_ratings == 0].sort_values(ascending=False)
    return recommendations.head(n_recommendations)

recommendations = recommend_mobiles('A', user_item_matrix)
print(f"\n📦 Recommended mobiles for User A:\n{recommendations}")
