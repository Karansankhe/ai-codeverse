import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier

# Load the dataset from a CSV file (replace with your file path)
df = pd.read_csv(r'C:\Users\Karan Sankhe\Desktop\ai\Network threat\cyberthreat.csv')

# Encode categorical features
data_encoded = pd.get_dummies(df, columns=['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address'],
                               prefix=['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address'])

X = data_encoded.drop("Target Variable", axis=1)
y = data_encoded["Target Variable"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an objective function for Optuna
def objective(trial):
    # Set up a pipeline for resampling and classification
    pipeline = Pipeline([
        ('sampling', RandomOverSampler()),  # You can try RandomUnderSampler for undersampling
        ('classification', KerasClassifier(build_fn=create_model(trial),
                                            epochs=trial.suggest_int('epochs', 10, 50),
                                            batch_size=trial.suggest_int('batch_size', 16, 64),
                                            verbose=0))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def create_model(trial):
    # Build a simple feedforward neural network
    model = keras.Sequential()
    model.add(layers.Dense(units=trial.suggest_int('units1', 32, 128), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(units=trial.suggest_int('units2', 32, 128), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Use softmax for multi-class classification

    model.compile(optimizer=trial.suggest_categorical('optimizer', ['adam', 'sgd']),
                  loss='binary_crossentropy',  # Use categorical_crossentropy for multi-class
                  metrics=['accuracy'])
    
    return model

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params

# Set up the final model with the best hyperparameters
final_model = create_model(best_params)

# Fit the final model
final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Make predictions on the test set
y_pred = (final_model.predict(X_test) > 0.5).astype("int32")  # Use threshold for binary classification

# Calculate accuracy and display metrics
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)
