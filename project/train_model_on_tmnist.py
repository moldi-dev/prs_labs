import pandas as pd
import os
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

CSV_FILE = "TMNIST_digits.csv"
MODEL_PATH = "ean13_digit_model.keras"
IMG_SIZE = (28, 28)
BATCH_SIZE = 64
EPOCHS = 30


def load_data():
    if not os.path.exists(CSV_FILE):
        print(f"[Error] '{CSV_FILE}' not found")
        exit(1)

    print("[Data] Loading CSV into memory...")
    df = pd.read_csv(CSV_FILE)

    # Extract Labels
    y = df['labels'].values

    # Extract Pixels
    # Drop 'names' and 'labels', keep only the 784 pixel columns
    # TMNIST columns usually start after 'labels'
    x = df.drop(columns=['names', 'labels'], errors='ignore').values

    # Reshape (N, 28, 28, 1) and then normalize
    x = x.reshape(-1, 28, 28, 1)
    x = x.astype('float32') / 255.0

    print(f"[Data] Loaded {len(x)} images.")

    # 90-10 train test split
    return train_test_split(x, y, test_size=0.1, random_state=42)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Layer 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Layer 3 (extra capacity for 2000+ fonts)
        layers.Conv2D(128, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevent overfitting on specific fonts
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Load
    X_train, X_test, y_train, y_test = load_data()

    print(f"[Training] Starting training on {len(X_train)} samples...")

    # Train
    model = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )

    # Save
    model.save(MODEL_PATH)
    print(f"\n[Success] Model saved to {MODEL_PATH}")