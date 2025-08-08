import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array
)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

here       = Path(__file__).parent
project    = here.parent
orig_root  = project / 'dataset'
split_root = project / 'dataset_split'
train_root = split_root / 'train'
val_root   = split_root / 'validation'
test_root  = split_root / 'test'

for d in (train_root, val_root, test_root):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

classes = [d for d in orig_root.iterdir() if d.is_dir() and d.name.isdigit()]

# Divisão em 70/15/15(treino, validação e teste)
print("Dividindo o dataset")
for class_dir in classes:
    imgs = list(class_dir.glob('*.*'))
    if not imgs:
        continue
    train_imgs, temp = train_test_split(imgs, test_size=0.3, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=seed)
    for subset_root, subset in (
        (train_root, train_imgs),
        (val_root,   val_imgs),
        (test_root,  test_imgs)
    ):
        dst = subset_root / class_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, dst / img.name)

print("Divisão concluída:",
      f"train={sum(1 for _ in train_root.rglob('*.*'))},",
      f"val={sum(1 for _ in val_root.rglob('*.*'))},",
      f"test={sum(1 for _ in test_root.rglob('*.*'))}")

# Definição da mediana
train_class_dirs = [d for d in train_root.iterdir() if d.is_dir()]
counts = {d.name: len(list(d.glob('*.*'))) for d in train_class_dirs}
target_count = int(np.median(list(counts.values())))
print(f"\nBalanceando para máximo de {target_count} imagens")

oversampler = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, fill_mode='nearest'
)

for cls_dir in train_class_dirs:
    imgs = list(cls_dir.glob('*.*'))
    n = len(imgs)
    if n > target_count:
        # Undersampling
        to_remove = random.sample(imgs, n - target_count)
        for f in to_remove:
            f.unlink()
        print(f"Classe {cls_dir.name}: removidas {n - target_count} imagens")
    elif n < target_count:
        # Oversampling
        needed = target_count - n
        gen = 0
        while gen < needed:
            for img_path in imgs:
                if gen >= needed:
                    break
                img = load_img(img_path, target_size=(32,32))
                x = img_to_array(img).reshape((1,32,32,3))
                for _ in oversampler.flow(
                    x, batch_size=1,
                    save_to_dir=str(cls_dir),
                    save_prefix=f"{cls_dir.name}_aug",
                    save_format='png'
                ):
                    gen += 1
                    if gen >= needed:
                        break
        print(f"Classe {cls_dir.name}: adicionadas {needed} imagens")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_root, target_size=(32,32), batch_size=32,
    class_mode='categorical', shuffle=True, seed=seed
)
val_gen = val_test_datagen.flow_from_directory(
    val_root, target_size=(32,32), batch_size=32,
    class_mode='categorical', shuffle=False, seed=seed
)
test_gen = val_test_datagen.flow_from_directory(
    test_root, target_size=(32,32), batch_size=32,
    class_mode='categorical', shuffle=False, seed=seed
)

# Definição da CNN
num_classes = len(train_gen.class_indices)
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treino com EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_gen, epochs=25, validation_data=val_gen, callbacks=[early_stopping])

# Gráficos de Acurácia e Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss de Treino e Validação')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Avaliação final
test_loss, test_acc = model.evaluate(test_gen)
print(f"\nTest loss: {test_loss:.4f} — Test accuracy: {test_acc:.4f}")

# Classification Report
y_true      = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred      = np.argmax(y_pred_prob, axis=1)
labels_str = sorted(train_gen.class_indices.keys(), key=lambda s: int(s))
labels_idx = [train_gen.class_indices[s] for s in labels_str]
print("\nClassification Report (0→24):")
print(classification_report(y_true, y_pred, labels=labels_idx, target_names=labels_str))

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_str)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation='vertical')
ax.set_xlabel("Rótulo Previsto")
ax.set_ylabel("Rótulo Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()
