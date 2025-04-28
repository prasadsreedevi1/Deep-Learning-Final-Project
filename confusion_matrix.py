from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from shiv_preprocessing import get_data
from cathy_model import CNN 

# true labels
test_imgs, test_genre_labels, test_va_labels = get_data('data/test_data_shiv.csv')
true_labels = test_genre_labels.numpy()

# predicted labels
cnn_model = CNN()
genre_logits, _ = cnn_model.call(test_imgs, is_testing=True)
predicted_labels = tf.argmax(genre_logits, axis=1, output_type=tf.int32).numpy()

# confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
genre_names = ["classical", "country", "jazz"]
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix: Genre Classification")
plt.show()