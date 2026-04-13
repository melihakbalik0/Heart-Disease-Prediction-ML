from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(df):

    # Features and Target separation
    X = df.drop('HeartDiseaseorAttack', axis=1)
    y = df['HeartDiseaseorAttack']

    # Stratified Split (80% Train / 20% Test)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling: Z-score Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training with Optimized Hyperparameters

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, X_test_scaled, y_test