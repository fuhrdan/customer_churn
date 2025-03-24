import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a sample dataset (for demonstration purposes)
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'churn_model.pkl')  # Save the model

# Load the trained model
model = joblib.load('churn_model.pkl')

# Create GUI application
class ChurnPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Churn Prediction")
        
        self.labels = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        self.entries = []
        
        for i, label in enumerate(self.labels):
            tk.Label(root, text=label).grid(row=i, column=0)
            entry = tk.Entry(root)
            entry.grid(row=i, column=1)
            self.entries.append(entry)
        
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_churn)
        self.predict_button.grid(row=len(self.labels), columnspan=2)
    
    def predict_churn(self):
        try:
            features = np.array([[float(entry.get()) for entry in self.entries]])
            prediction = model.predict(features)[0]
            result = "Churn" if prediction == 1 else "No Churn"
            messagebox.showinfo("Prediction Result", f"Customer will: {result}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for all features.")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictionApp(root)
    root.mainloop()
