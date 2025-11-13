# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import time
import tracemalloc

# 🎨 Fix backend for matplotlib in certain IDEs
import matplotlib
matplotlib.use("TkAgg")

# 1. Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 2. Load preprocessed data
X = np.load("X_combined.npy")  # Input features (text + sentiment)
y = np.load("y.npy")           # Labels (0 or 1)

print("Loaded X shape:", X.shape)
print("Loaded y shape:", y.shape)

# 3. Generate dummy PHQ-8 inputs (can be replaced with actual questionnaire data)
phq_fake = np.random.randint(0, 4, size=(X.shape[0], 8))  # 8 questions, score 0-3
print("🧪 Generated fake PHQ-8 input with shape:", phq_fake.shape)

# 4. Split into training, validation, and test sets (70/15/15)
X_train, X_temp, y_train, y_temp, phq_train, phq_temp = train_test_split(
    X, y, phq_fake, test_size=0.3, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test, phq_val, phq_test = train_test_split(
    X_temp, y_temp, phq_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

# Display label distributions
print("\n Label Distribution:")
print("Train:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Val:  ", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test: ", dict(zip(*np.unique(y_test, return_counts=True))))

# 5. Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\n Class Weights:", class_weights)

# 6. Load tokenizer and build GloVe embedding matrix
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

vocab_size = 10000
embedding_dim = 100
max_length = 300

# Load GloVe word embeddings
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# Build embedding matrix aligned with tokenizer indices
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size and word in embedding_index:
        embedding_matrix[i] = embedding_index[word]

# Count how many tokens had GloVe vectors
found = sum((embedding_matrix != 0).any(axis=1))
print(f"{found}/{vocab_size} words initialized with GloVe vectors")

# 7. Separate text and sentiment input from combined X
X_train_text = X_train[:, :-1]
X_train_sent = X_train[:, -1].reshape(-1, 1)
X_val_text = X_val[:, :-1]
X_val_sent = X_val[:, -1].reshape(-1, 1)
X_test_text = X_test[:, :-1]
X_test_sent = X_test[:, -1].reshape(-1, 1)

# 8. Build LSTM model with three inputs: text, sentiment, PHQ
# Text input and embedding
text_input = tf.keras.layers.Input(shape=(max_length,), name="text_input")
x = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True
)(text_input)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.2))(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Sentiment score input
sentiment_input = tf.keras.layers.Input(shape=(1,), name="sentiment_input")
sentiment_dense = tf.keras.layers.Dense(16, activation="relu")(sentiment_input)

# PHQ input
phq_input = tf.keras.layers.Input(shape=(8,), name="phq_input")
phq_dense = tf.keras.layers.Dense(16, activation="relu")(phq_input)

# Merge all inputs
combined = tf.keras.layers.Concatenate()([x, sentiment_dense, phq_dense])
combined = tf.keras.layers.Dense(32, activation="relu")(combined)
output = tf.keras.layers.Dense(1, activation="sigmoid")(combined)

# Define and compile model
model = tf.keras.models.Model(
    inputs={"text_input": text_input, "sentiment_input": sentiment_input, "phq_input": phq_input},
    outputs=output
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 9. Define callbacks for early stopping and learning rate reduction
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# Start memory and time tracking
tracemalloc.start()
start_time = time.time()

# 10. Train the model
history = model.fit(
    {"text_input": X_train_text, "sentiment_input": X_train_sent, "phq_input": phq_train},
    y_train,
    validation_data=(
        {"text_input": X_val_text, "sentiment_input": X_val_sent, "phq_input": phq_val},
        y_val
    ),
    epochs=200,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# End timing and print memory usage
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Training Time: {end_time - start_time:.2f} seconds")
print(f"Peak Memory Used During Training: {peak / (1024 * 1024):.2f} MB")

# 10.1 Plot loss and accuracy over epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("Saved training_history.png")

# 🧪 11. Evaluate the model on the test set
y_pred_prob = model.predict({
    "text_input": X_test_text,
    "sentiment_input": X_test_sent,
    "phq_input": phq_test
})
y_pred = (y_pred_prob > 0.5).astype("int32")  # Convert probabilities to class labels

# Print accuracy and classification metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_phq.png")
print("Saved confusion_matrix_phq.png")

# Additional evaluation: ROC curve and AUC
from sklearn.metrics import roc_auc_score, roc_curve

roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("Saved roc_curve.png")

# 12. Save the trained model
model.save("lstm_depression_model_with_phq.keras")
print("✅ Model saved as lstm_depression_model_with_phq.keras")
