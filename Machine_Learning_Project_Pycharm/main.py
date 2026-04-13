from src.data_loader import load_heart_data
from src.model_trainer import train_random_forest
from src.visualizer import (
    save_performance_plots,
    plot_data_quality,
    plot_roc_curve,
    plot_precision_recall,
    plot_correlation_heatmap
)
from sklearn.metrics import classification_report


def main():
    print("--- Heart Disease Prediction System Initialized ---")
    df = load_heart_data("data/heart.csv")

    plot_data_quality(df)
    plot_correlation_heatmap(df)

    model, X_test, y_test = train_random_forest(df)

    predictions = model.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, predictions))

    feature_names = df.drop('HeartDiseaseorAttack', axis=1).columns
    save_performance_plots(model, X_test, y_test, feature_names)
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall(model, X_test, y_test)

    print("\nExecution completed. All reports are ready.")


if __name__ == "__main__":
    main()