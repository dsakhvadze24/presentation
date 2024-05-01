import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Step 1: Import and review dataset
data = pd.read_csv("dataset.csv")
print(data.head())

# Step 2: Remove individual and redundant features
features_to_remove = ['URL', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']
data.drop(features_to_remove, axis=1, inplace=True)

# Step 3: Encode categorical data
data = pd.get_dummies(data, columns=['CHARSET', 'SERVER', 'WHOIS_COUNTRY'])

# Step 4: Handle missing values
data.fillna(data.median(), inplace=True)

# Step 5: Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

# Step 6: Split dataset into training and test datasets
from sklearn.model_selection import train_test_split

X = data.drop('Type', axis=1)
y = data['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Define model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 8: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the model on training dataset
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 10: Use model to validate test dataset
y_pred = model.predict(X_test)

# Step 11: Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Step 12: Analyze test dataset using ROC/AUC, confusion matrix, predicting probability for test set, classification report

# ROC/AUC
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC: {roc_auc}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred.round())
print('Confusion Matrix:')
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred.round())
print('Classification Report:')
print(class_report)

# Step 13: Build plots illustrating analysis
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

