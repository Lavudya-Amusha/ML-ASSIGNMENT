import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load the Titanic dataset
@st.cache
def load_data():
    return sns.load_dataset('titanic')

df = load_data()

# Title and Introduction
st.title("Titanic Dataset Machine Learning Models")
st.write("This dashboard allows you to apply various machine learning models to the Titanic dataset and compare their performance.")

# Display the Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocess the Data
st.subheader("Preprocessing the Data")
st.write("Selecting relevant features and handling missing values...")

# Feature Selection and Encoding
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df.dropna(inplace=True)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

st.write("Processed dataset:")
st.dataframe(df.head())

# Split Data
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
st.subheader("Choose a Machine Learning Model")
model_choice = st.selectbox(
    "Select a model:",
    ["Linear Regression", "Multiple Linear Regression", "Logistic Regression", 
     "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", 
     "Support Vector Machine (SVM)", "Naive Bayes", "K-Means Clustering"]
)

# Train and Evaluate Function
def train_and_evaluate(model, regression=False):
    if regression:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, report

# Model Training
if st.button("Train Model"):
    st.write(f"### {model_choice} Results")

    if model_choice == "Linear Regression":
        model = LinearRegression()
        mse, r2 = train_and_evaluate(model, regression=True)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

    elif model_choice == "Multiple Linear Regression":
        model = LinearRegression()
        mse, r2 = train_and_evaluate(model, regression=True)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "Support Vector Machine (SVM)":
        model = SVC()
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "Naive Bayes":
        model = GaussianNB()
        accuracy, report = train_and_evaluate(model)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    elif model_choice == "K-Means Clustering":
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)
        df['cluster'] = kmeans.labels_
        st.write("Cluster assignment added to the dataset:")
        st.dataframe(df[['survived', 'cluster']].head())

# Visualizations
st.subheader("Visualizations")
st.write("Correlation Heatmap:")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("Survival Count:")
fig, ax = plt.subplots()
sns.countplot(data=df, x='survived', palette='viridis', ax=ax)
st.pyplot(fig)

