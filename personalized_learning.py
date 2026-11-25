# ============================================================
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# MACHINE LEARNING PBL PROJECT
# Dataset: Kaggle - Student Performance Factors
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CLASS
# ============================================================

class PersonalizedLearningRecommendationSystem:

    def __init__(self, filename):

        self.filename = filename
        self.df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.preprocessor = None
        self.model = None
        self.predictions = None

        self.feature_columns = []
        self.recommendation_history = []

        # Create an output folder beside this Python file.
        # If a file named "outputs" already exists, use "outputs_folder"
        # instead of deleting or overwriting the user's file.
        project_directory = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(project_directory, "outputs")

        if os.path.isfile(self.output_folder):
            self.output_folder = os.path.join(
                project_directory, "outputs_folder"
            )

        os.makedirs(self.output_folder, exist_ok=True)

    # ========================================================
    # 3. LOAD DATASET
    # ========================================================

    def load_dataset(self):

        try:
            self.df = pd.read_csv(self.filename)
            self.df.columns = self.df.columns.astype(str).str.strip()

            print("\n" + "=" * 70)
            print("DATASET LOADED SUCCESSFULLY")
            print("=" * 70)

            print("\nFirst 5 Rows:")
            print(self.df.head())

            print("\nDataset loaded successfully.")
            print("Dataset shape:", self.df.shape)

        except FileNotFoundError:
            print("\nError: Dataset file not found.")
            print("Place student_performance.csv in the same folder as this Python file.")

    # ========================================================
    # 4. BASIC INFORMATION
    # ========================================================

    def basic_information(self):

        if self.df is None:
            print("\nPlease load the dataset first.")
            return

        print("\n" + "=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)

        print("\nDataset Shape:")
        print(self.df.shape)

        print("\nColumn Names:")
        print(self.df.columns.tolist())

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nDataset Information:")
        self.df.info()

    # ========================================================
    # 5. STATISTICAL SUMMARY
    # ========================================================

    def statistical_summary(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("STATISTICAL SUMMARY")
        print("=" * 70)

        print(self.df.describe(include="all").transpose())

    # ========================================================
    # 6. CHECK MISSING VALUES
    # ========================================================

    def check_missing_values(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("MISSING VALUES")
        print("=" * 70)

        print("\nMissing values:")
        print(self.df.isnull().sum())

        print("\nMissing Value Percentage:")
        print((self.df.isnull().mean() * 100).round(2))

    # ========================================================
    # 7. HANDLE MISSING VALUES
    # ========================================================

    def handle_missing_values(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("HANDLING MISSING VALUES")
        print("=" * 70)

        # The model pipeline also handles missing values safely.
        # Here we fill missing values for EDA and printed dataset inspection.
        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        categorical_columns = self.df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in numerical_columns:
            if self.df[column].isnull().sum() > 0:
                median_value = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_value)
                print(column, "-> Missing values filled using Median")

        for column in categorical_columns:
            if self.df[column].isnull().sum() > 0:
                modes = self.df[column].mode()
                if not modes.empty:
                    self.df[column] = self.df[column].fillna(modes.iloc[0])
                    print(column, "-> Missing values filled using Mode")

        print("\nMissing values after treatment:")
        print(self.df.isnull().sum())

    # ========================================================
    # 8. CHECK DUPLICATES
    # ========================================================

    def handle_duplicates(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("DUPLICATE RECORDS")
        print("=" * 70)

        duplicate_count = self.df.duplicated().sum()
        print("Number of duplicate rows:", duplicate_count)

        if duplicate_count > 0:
            self.df = self.df.drop_duplicates().copy()
            print("Duplicate rows removed successfully.")
        else:
            print("No duplicate rows found.")

    # ========================================================
    # 9. CHECK CATEGORICAL VALUES
    # ========================================================

    def categorical_values(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("CATEGORICAL VALUES")
        print("=" * 70)

        categorical_columns = self.df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in categorical_columns:
            print("\nColumn:", column)
            print(self.df[column].value_counts(dropna=False))

    # ========================================================
    # 10. OUTLIER DETECTION USING IQR
    # ========================================================

    def detect_outliers(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("OUTLIER DETECTION USING IQR")
        print("=" * 70)

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numerical_columns:
            values = self.df[column].dropna()

            if values.empty:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = values[
                (values < lower_bound) | (values > upper_bound)
            ]

            print("\nColumn:", column)
            print("Q1:", round(q1, 2))
            print("Q3:", round(q3, 2))
            print("IQR:", round(iqr, 2))
            print("Number of outliers:", len(outliers))

    # ========================================================
    # 11. BOXPLOT
    # ========================================================

    def display_boxplots(self):

        if self.df is None:
            print("\nPlease load the dataset first.")
            return

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        if len(numerical_columns) == 0:
            print("No numerical columns available for boxplots.")
            return

        print("\n" + "=" * 70)
        print("DISPLAYING BOXPLOTS")
        print("=" * 70)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=self.df[numerical_columns])
        plt.title("Box Plot of Numerical Student Variables")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "boxplots.png"))
        plt.show()
        plt.close()

    # ========================================================
    # 12. HISTOGRAM
    # ========================================================

    def display_histograms(self):

        if self.df is None:
            return

        numerical_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        if len(numerical_columns) == 0:
            return

        print("\n" + "=" * 70)
        print("DISPLAYING HISTOGRAMS")
        print("=" * 70)

        self.df[numerical_columns].hist(
            figsize=(14, 10),
            bins=20,
            edgecolor="black"
        )

        plt.suptitle("Distribution of Numerical Student Variables")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "histograms.png"))
        plt.show()
        plt.close()

    # ========================================================
    # 13. CORRELATION ANALYSIS
    # ========================================================

    def correlation_analysis(self):

        if self.df is None:
            return

        numerical_data = self.df.select_dtypes(include=np.number)

        if numerical_data.shape[1] < 2:
            print("Not enough numerical columns for correlation.")
            return

        correlation = numerical_data.corr()

        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)

        if "Exam_Score" in correlation.columns:
            print("\nCorrelation with Exam_Score:")
            print(correlation["Exam_Score"].sort_values(ascending=False))

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "correlation_heatmap.png"))
        plt.show()
        plt.close()

    # ========================================================
    # 14. EDA GRAPHS
    # ========================================================

    def perform_eda(self):

        if self.df is None:
            return

        print("\n" + "=" * 70)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 70)

        target_column = getattr(self, "target_column", None)
        if target_column is None:
            target_column = next(
                (c for c in ["Exam_Score", "Performance Index", "Performance",
                             "GPA", "Score", "Final_Score", "G3"]
                 if c in self.df.columns),
                None
            )

        possible_inputs = [
            ("Hours_Studied", "Hours Studied"),
            ("Hours Studied", "Hours Studied"),
            ("Attendance", "Attendance"),
            ("AttendanceRate", "Attendance"),
            ("Previous_Scores", "Previous Scores"),
            ("Previous Scores", "Previous Scores")
        ]
        graph_pairs = [
            (column, target_column, f"{label} vs {target_column}")
            for column, label in possible_inputs
            if column in self.df.columns and target_column in self.df.columns
        ]

        for x_column, y_column, title in graph_pairs:
            if x_column in self.df.columns and y_column in self.df.columns:
                plt.figure(figsize=(8, 5))
                sns.scatterplot(
                    data=self.df,
                    x=x_column,
                    y=y_column
                )
                plt.title(title)
                plt.tight_layout()
                filename = title.lower().replace(" ", "_") + ".png"
                plt.savefig(os.path.join(self.output_folder, filename))
                plt.show()
                plt.close()
            else:
                print("Skipped graph because a required column is missing:", title)

        print("EDA graphs saved in the outputs folder.")

    # ========================================================
    # 15. PREPARE FEATURES
    # ========================================================

    def prepare_features(self):

        if self.df is None:
            print("Please load the dataset first.")
            return

        print("\\n" + "=" * 70)
        print("FEATURE SELECTION")
        print("=" * 70)

        # Support common target-column names used by different student datasets.
        # Column matching ignores spaces, underscores, and capital letters.
        def normalize_name(name):
            return "".join(ch.lower() for ch in str(name) if ch.isalnum())

        target_candidates = [
            "Exam_Score", "Exam Score", "Performance Index",
            "Performance", "Final_Score", "Final Score",
            "Final_Grade", "Final Grade", "GPA", "Grade",
            "Score", "Student_Performance", "Student Performance",
            "Overall_Score", "Overall Score", "Result", "G3"
        ]

        normalized_columns = {
            normalize_name(column): column for column in self.df.columns
        }

        target_column = None
        for candidate in target_candidates:
            match = normalized_columns.get(normalize_name(candidate))
            if match is not None:
                # Target must be numeric or convertible to numeric.
                converted = pd.to_numeric(self.df[match], errors="coerce")
                if converted.notna().sum() > 0:
                    target_column = match
                    break

        if target_column is None:
            print("\\nCould not automatically identify the score/target column.")
            print("Available dataset columns:")
            print(self.df.columns.tolist())
            print("\\nPlease rename the column you want to predict to 'Exam_Score',")
            print("or add its name to target_candidates in prepare_features().")
            raise ValueError(
                "Target column not found. See the printed column names above."
            )

        # Use all other useful columns as features, excluding likely ID columns.
        excluded_names = {
            normalize_name(target_column), "studentid", "id", "studentnumber",
            "rownumber", "index"
        }
        selected_features = [
            column for column in self.df.columns
            if normalize_name(column) not in excluded_names
            and self.df[column].nunique(dropna=True) > 1
        ]

        if not selected_features:
            raise ValueError(
                "No usable feature columns were found in the dataset."
            )

        self.target_column = target_column
        self.feature_columns = selected_features
        self.X = self.df[selected_features].copy()
        self.y = pd.to_numeric(self.df[target_column], errors="coerce")

        # Some datasets store categories such as Gender as numbers (for example,
        # 0/1). Treat these as categories so users can enter values like female.
        categorical_name_parts = (
            "gender", "sex", "schooltype", "internet", "motivation",
            "resources", "extracurricular", "parental", "peerinfluence",
            "familyincome", "learningstyle", "preference", "category",
            "level", "status", "subject", "course", "major"
        )
        for column in self.X.columns:
            normalized_column = normalize_name(column)
            if (not pd.api.types.is_numeric_dtype(self.X[column]) or
                    any(part in normalized_column for part in categorical_name_parts)):
                self.X[column] = self.X[column].astype("string")

        # Keep the feature list in sync with the prepared model inputs.
        self.feature_columns = self.X.columns.tolist()

        valid_rows = self.y.notna()
        self.X = self.X.loc[valid_rows].copy()
        self.y = self.y.loc[valid_rows].copy()

        if len(self.X) < 5:
            raise ValueError(
                "The dataset has fewer than 5 usable rows after checking the target column."
            )

        print("\\nTarget Variable:", target_column)
        print("\\nSelected Features:")
        for feature in selected_features:
            print("-", feature)

        print("\\nRows available for model:", len(self.X))

    # ========================================================
    # 16. TRAIN TEST SPLIT
    # ========================================================

    def split_data(self):

        if self.X is None or self.y is None:
            print("Please prepare features first.")
            return

        print("\n" + "=" * 70)
        print("TRAINING AND TESTING SPLIT")
        print("=" * 70)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.20,
                random_state=42
            )
        )

        print("Training records:", len(self.X_train))
        print("Testing records:", len(self.X_test))

    # ========================================================
    # 17. CREATE PREPROCESSING PIPELINE
    # ========================================================

    def create_preprocessor(self):

        numerical_features = self.X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = [
            column for column in self.X.columns
            if column not in numerical_features
        ]

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []

        if numerical_features:
            transformers.append(
                ("numerical", numerical_pipeline, numerical_features)
            )

        if categorical_features:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return self.preprocessor

    # ========================================================
    # 18. TRAIN RANDOM FOREST MODEL
    # ========================================================

    def train_model(self):

        print("\n" + "=" * 70)
        print("RANDOM FOREST REGRESSION MODEL")
        print("=" * 70)

        preprocessor = self.create_preprocessor()

        self.model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ))
        ])

        self.model.fit(self.X_train, self.y_train)

        self.predictions = self.model.predict(self.X_test)

        self.evaluate_model(
            self.y_test,
            self.predictions,
            "Random Forest Regression"
        )

        return self.predictions

    # ========================================================
    # 19. MODEL EVALUATION
    # ========================================================

    def evaluate_model(self, actual, predicted, model_name):

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)

        print("\nModel:", model_name)
        print("Mean Absolute Error:", round(mae, 2))
        print("Mean Squared Error:", round(mse, 2))
        print("Root Mean Squared Error:", round(rmse, 2))
        print("R2 Score:", round(r2, 4))

        metrics = pd.DataFrame([{
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2
        }])

        metrics.to_csv(
            os.path.join(self.output_folder, "model_evaluation.csv"),
            index=False
        )

    # ========================================================
    # 20. ACTUAL VS PREDICTED GRAPH
    # ========================================================

    def prediction_graph(self, predictions):

        plt.figure(figsize=(8, 5))
        plt.scatter(self.y_test, predictions, alpha=0.6)

        minimum = min(self.y_test.min(), predictions.min())
        maximum = max(self.y_test.max(), predictions.max())

        plt.plot([minimum, maximum], [minimum, maximum])
        target_label = getattr(self, "target_column", "Predicted Score")
        plt.title(f"Actual vs Predicted {target_label}")
        plt.xlabel(f"Actual {target_label}")
        plt.ylabel(f"Predicted {target_label}")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.output_folder, "actual_vs_predicted.png")
        )
        plt.show()
        plt.close()

    # ========================================================
    # 21. FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(self):

        if self.model is None:
            print("Please train the model first.")
            return

        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE")
        print("=" * 70)

        trained_model = self.model.named_steps["model"]
        preprocessor = self.model.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()
        importance = trained_model.feature_importances_

        importance_data = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        print(importance_data.head(15).to_string(index=False))

        importance_data.to_csv(
            os.path.join(self.output_folder, "feature_importance.csv"),
            index=False
        )

        top_features = importance_data.head(10).sort_values(
            by="Importance"
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_features,
            x="Importance",
            y="Feature"
        )
        plt.title("Top Important Features")
        plt.tight_layout()

        plt.savefig(
            os.path.join(self.output_folder, "feature_importance.png")
        )
        plt.show()
        plt.close()

    # ========================================================
    # 22. PERSONALIZED RESOURCE RECOMMENDATIONS
    # ========================================================

    def recommend_resources(self, predicted_score, topic, learning_preference):

        topic = topic.strip().title()
        preference = learning_preference.strip().lower()

        print("\n" + "=" * 70)
        print("PERSONALIZED LEARNING RECOMMENDATIONS")
        print("=" * 70)

        print("Selected topic:", topic)
        print("Learning preference:", learning_preference)
        print("Predicted exam score:", round(predicted_score, 2))

        # Recommend resource types according to the student's preference.
        if preference == "visual":
            resources = [
                ("Video lesson", f"Watch a step-by-step video about {topic}."),
                ("Diagram / mind map", f"Make a diagram or mind map for {topic}."),
                ("Infographic", f"Review a visual summary of {topic}.")
            ]

        elif preference == "auditory":
            resources = [
                ("Audio lesson", f"Listen to a lesson explaining {topic}."),
                ("Group discussion", f"Discuss {topic} with classmates."),
                ("Teach-back", f"Explain {topic} aloud in your own words.")
            ]

        elif preference == "practical":
            resources = [
                ("Hands-on activity", f"Complete a practical task about {topic}."),
                ("Worked examples", f"Solve examples on {topic}."),
                ("Mini-project", f"Build a small project using {topic}.")
            ]

        else:
            resources = [
                ("Study notes", f"Read structured notes about {topic}."),
                ("Flashcards", f"Create question-and-answer cards for {topic}."),
                ("Practice quiz", f"Attempt a short quiz on {topic}.")
            ]

        # Use predicted performance to suggest an appropriate level.
        if predicted_score < 60:
            resources.insert(
                0,
                (
                    "Beginner revision",
                    f"Revise the basic concepts of {topic} before moving to advanced questions."
                )
            )
            performance_level = "Needs additional practice"

        elif predicted_score < 75:
            resources.insert(
                0,
                (
                    "Guided practice",
                    f"Study worked examples and solve medium-level questions on {topic}."
                )
            )
            performance_level = "Developing"

        else:
            resources.append(
                (
                    "Challenge activity",
                    f"Try advanced questions or a mini-project related to {topic}."
                )
            )
            performance_level = "Ready for extension practice"

        print("Suggested study level:", performance_level)
        print("\nRecommended resources:")

        for number, (resource_type, description) in enumerate(resources, start=1):
            print(f"{number}. {resource_type}: {description}")

            self.recommendation_history.append({
                "Topic": topic,
                "Learning_Preference": learning_preference,
                "Predicted_Exam_Score": round(float(predicted_score), 2),
                "Performance_Level": performance_level,
                "Resource_Type": resource_type,
                "Recommendation": description
            })

        output_file = os.path.join(
            self.output_folder,
            "learning_recommendations.csv"
        )

        pd.DataFrame(self.recommendation_history).to_csv(
            output_file,
            index=False
        )

        print("\nRecommendations saved to:", output_file)

    # ========================================================
    # 23. INPUT NEW STUDENT AND GENERATE RECOMMENDATIONS
    # ========================================================

    def predict_new_student(self):

        print("\n" + "=" * 70)
        print("PERSONALIZED LEARNING SYSTEM - NEW STUDENT")
        print("=" * 70)

        try:
            student_values = {}

            for column in self.feature_columns:
                if pd.api.types.is_numeric_dtype(self.X[column]):
                    while True:
                        raw_value = input(f"Enter {column} (number): ").strip()
                        try:
                            student_values[column] = float(raw_value)
                            break
                        except ValueError:
                            print("Please enter a valid number for", column)
                else:
                    # Show known values when available to help match training data.
                    known_values = self.X[column].dropna().astype(str).unique().tolist()
                    if 0 < len(known_values) <= 12:
                        print(f"Possible values for {column}: {', '.join(known_values)}")
                    student_values[column] = input(
                        f"Enter {column} (text/category): "
                    ).strip()

            for column, value in student_values.items():
                normalized = "".join(ch.lower() for ch in column if ch.isalnum())
                if "hour" in normalized and value < 0:
                    raise ValueError(f"{column} cannot be negative.")
                if ("attendance" in normalized or "score" in normalized
                        or normalized in {"gpa", "performanceindex"}):
                    if value < 0:
                        raise ValueError(f"{column} cannot be negative.")

            new_student = pd.DataFrame(
                [student_values],
                columns=self.X.columns
            )

            predicted_score = self.model.predict(new_student)[0]

            print("\nPredicted Exam Score:", round(predicted_score, 2))

            topic = input(
                "Enter topic you want to learn (e.g. Python, DBMS, Networking): "
            ).strip()

            if not topic:
                raise ValueError("Topic cannot be empty.")

            print("\nChoose your learning preference:")
            print("1. Visual (videos, diagrams, images)")
            print("2. Auditory (audio, discussion)")
            print("3. Practical (activities, projects)")
            print("4. Reading (notes, flashcards)")

            preference_choice = input("Enter choice (1-4): ").strip()

            preference_map = {
                "1": "Visual",
                "2": "Auditory",
                "3": "Practical",
                "4": "Reading"
            }

            if preference_choice not in preference_map:
                raise ValueError("Please choose a preference from 1 to 4.")

            preference = preference_map[preference_choice]

            self.recommend_resources(
                predicted_score,
                topic,
                preference
            )

        except ValueError as error:
            print("\nInvalid input:", error)
            print("Please run the program again and enter valid values.")

        except Exception as error:
            print("\nAn error occurred:", error)
            print("Check the input fields and dataset columns.")

    # ========================================================
    # 24. MAIN RUN METHOD
    # ========================================================

    def run(self):

        self.load_dataset()

        if self.df is None:
            return

        self.basic_information()
        self.statistical_summary()
        self.check_missing_values()
        self.handle_missing_values()
        self.handle_duplicates()
        self.categorical_values()
        self.detect_outliers()
        self.display_boxplots()
        self.display_histograms()
        self.correlation_analysis()
        self.perform_eda()

        self.prepare_features()
        self.split_data()
        self.train_model()
        self.prediction_graph(self.predictions)
        self.feature_importance()
        self.predict_new_student()

        print("\n" + "=" * 70)
        print("MODEL TRAINING AND ANALYSIS COMPLETED")
        print("=" * 70)
        print("Graphs and metrics are saved in:", self.output_folder)
        print("If valid new-student inputs were entered, recommendations are saved in learning_recommendations.csv.")


# ============================================================
# 25. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("     PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
    print("                MACHINE LEARNING PBL")
    print("=" * 70)

    dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_performance.csv"
)
    

    project = PersonalizedLearningRecommendationSystem(
        dataset_path
    )

    project.run()


# ============================================================
# 26. RUN PROJECT
# ============================================================

if __name__ == "__main__":
    main()
