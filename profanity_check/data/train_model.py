"""Train Model from data"""
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

df = pd.concat([
    pd.read_csv("./jigsaw-toxic-comment-train_en_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_ru_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_fr_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_pt_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_es_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_it_clean.csv"),
    pd.read_csv("./jigsaw-toxic-comment-train_tr_clean.csv"),
])

texts = df["comment_text"].astype(str)
y = (df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].sum(axis=1) > 0).astype(int)

vectorizer = TfidfVectorizer(min_df=0.00001)
X = vectorizer.fit_transform(texts)

model = LinearSVC(
    class_weight="balanced",
    dual=False,
    tol=0.001,
    max_iter=1000)

#parameters = {
#    'C': [0.1, 1.0], 
#    'tol': [1e-1, 1e-2, 1e-3], 
#    'max_iter': [500, 1000, 10000]
#}
#
#clf = GridSearchCV(model, parameters, n_jobs=-1)
#clf.fit(X, y)
#calibrated_classifier_cv = CalibratedClassifierCV(estimator=clf.best_estimator_)

calibrated_classifier_cv = CalibratedClassifierCV(estimator=model)
calibrated_classifier_cv.fit(X, y)

dump(vectorizer, "vectorizer.joblib")
dump(calibrated_classifier_cv, "model.joblib")

result = calibrated_classifier_cv.score(X, y)
print("training accuracy: %.2f%%" % (result * 100.0))

# validate
df_validate = pd.concat([
    pd.read_csv("./validation_en.csv"),
])

texts_validate = df_validate["comment_text"].astype(str)
y_validate = df_validate['toxic']
X_validate = vectorizer.transform(texts_validate)

result = calibrated_classifier_cv.score(X_validate, y_validate)
print("svm validation accuracy: %.2f%%" % (result * 100.0))
