# Ensure the required libraries are installed
!pip install tensorflow scikit-learn imbalanced-learn optuna

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import optuna
from tensorflow import keras
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasClassifier  # Correct import
import joblib  # To save the model as .pkl

# Load the dataset from a CSV file
df = pd.read_csv('cyberthreat.csv')  # Adjust the path as needed

# Encode categorical features
data_encoded = pd.get_dummies(df, columns=['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address'],
                               prefix=['Protocol', 'Flag', 'Packet', 'Source IP Address', 'Destination IP Address'])

X = data_encoded.drop("Target Variable", axis=1)
y = data_encoded["Target Variable"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    pipeline = Pipeline([
        ('sampling', RandomOverSampler()),
        ('classification', KerasClassifier(build_fn=create_model(trial),
                                            epochs=trial.suggest_int('epochs', 10, 50),
                                            batch_size=trial.suggest_int('batch_size', 16, 64),
                                            verbose=0))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def create_model(trial):
    model = keras.Sequential()
    model.add(layers.Dense(units=trial.suggest_int('units1', 32, 128), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(units=trial.suggest_int('units2', 32, 128), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=trial.suggest_categorical('optimizer', ['adam', 'sgd']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Optimize the model
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters and create the final model
best_params = study.best_params
final_model = create_model(best_params)
final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Save the model as a .pkl file
joblib.dump(final_model, 'final_model.pkl')
