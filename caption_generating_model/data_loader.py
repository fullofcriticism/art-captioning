from sklearn.preprocessing import MultiLabelBinarizer
import csv

def load_classes(classes_path):
    """Загрузка классов из файла"""
    with open(classes_path) as load_file:
        reader = csv.reader(load_file, delimiter="\n")
        data = [tuple(row) for row in reader]
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(data)
    return binarizer