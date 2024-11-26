import os, numpy as np, cv2, tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef
from train import load_dataset, data_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

seed = 0

height = 256
width = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_memory():
    filePath = os.path.join("files", "memory.txt")
    f = open(filePath, 'r')
    return f.readlines()

def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((height, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    
    memory = read_memory()
    seed = int(memory[0])
    epochs = int(memory[1])
    np.random.seed(seed)
    tf.random.set_seed(seed)

    create_dir("results")

    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_dataset(data_path)

    print(data_path)
    

    SCORE = []
    for x, y in tqdm(zip(X_test, y_test), total=len(y_test)):
        name = x.split("\\")[-1]
        
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (width, height))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height))

        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        image = cv2.putText(image, text="Seed: "+str(seed)+" Num_Epochs: "+str(epochs),org=(10,30),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        image = cv2.putText(image, text="Image from Dataset",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        mask = cv2.putText(mask, text="Mask from Dataset",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        y_pred = cv2.putText(y_pred, text="Predicted Segmentation",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)


        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)
        
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()