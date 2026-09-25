from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


print("Loading MNIST dataset...")

# Download MNIST
mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset loaded.")
print("Number of images:", len(X))


# Normalize pixel values
X = X / 255.0


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)


print("Training model...")

# Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(128,),
    max_iter=20,
    random_state=42,
    verbose=True
)


# Train
model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy * 100:.2f}%")


# Create models folder if needed
os.makedirs("models", exist_ok=True)


# Save model
joblib.dump(
    model,
    "models/digit_model.pkl"
)

print("Model saved to models/digit_model.pkl")