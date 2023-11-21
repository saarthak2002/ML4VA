import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import pandas as pd

data = pd.read_csv('./Crash_Data.csv')

# fill in the columns where crash is not work zone related
data['WORK_ZONE_LOCATION'].fillna('Not Applicable', inplace=True)
data['WORK_ZONE_TYPE'].fillna('Not Applicable', inplace=True)

# Drop unnecessary columns with missing data
columns_to_drop = ['SPEED_DIFF_MAX', 'FUN', 'FAC', 'MPO_NAME', 'NODE', 'OFFSET', 'RNS_MP', 'OBJECTID', 'DOCUMENT_NBR', 'PHYSICAL_JURIS', 'RTE_NM', 'VSP', 'JURIS_CODE', 'K_PEOPLE', 'A_PEOPLE', 'B_PEOPLE', 'C_PEOPLE', 'PERSONS_INJURED']

data.drop(columns=columns_to_drop, inplace=True)

data.dropna(subset=['X', 'Y'], inplace=True)
data = data.reset_index(drop=True)

# encode date
from datetime import datetime

# 2015/10/16 03:59:59+00	-> number
def date_string_convert(date_string):
    length = len(date_string)
    date_string = date_string[0:length-3]
    date_obj = datetime.strptime(date_string, "%Y/%m/%d %H:%M:%S")
    timestamp = date_obj.timestamp()
    return timestamp

data["CRASH_DT"] = data["CRASH_DT"].apply(date_string_convert)

X = data.drop(columns=['CRASH_SEVERITY'])
y = data['CRASH_SEVERITY'].copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)



num_features = ["X", "Y", "CRASH_YEAR", "CRASH_DT", "CRASH_MILITARY_TM", "PEDESTRIANS_KILLED", "VEH_COUNT", "PEDESTRIANS_INJURED"]

one_hot_features = ["COLLISION_TYPE","WEATHER_CONDITION","LIGHT_CONDITION","ROADWAY_SURFACE_COND","RELATION_TO_ROADWAY","ROADWAY_ALIGNMENT", "ROADWAY_SURFACE_TYPE", "ROADWAY_DEFECT","ROADWAY_DESCRIPTION","INTERSECTION_TYPE","TRAFFIC_CONTROL_TYPE","TRFC_CTRL_STATUS_TYPE", "WORK_ZONE_RELATED","WORK_ZONE_LOCATION", "WORK_ZONE_TYPE","SCHOOL_ZONE","FIRST_HARMFUL_EVENT","FIRST_HARMFUL_EVENT_LOC","ALCOHOL_NOTALCOHOL", "ANIMAL","BELTED_UNBELTED","BIKE_NONBIKE", "DISTRACTED_NOTDISTRACTED","DROWSY_NOTDROWSY","DRUG_NODRUG","GR_NOGR","HITRUN_NOT_HITRUN", "LGTRUCK_NONLGTRUCK", "MOTOR_NONMOTOR","PED_NONPED","SPEED_NOTSPEED", "RD_TYPE","INTERSECTION_ANALYSIS","SENIOR_NOTSENIOR","YOUNG_NOTYOUNG", "MAINLINE_YN","NIGHT","VDOT_DISTRICT","AREA_TYPE","SYSTEM","OWNERSHIP","PLAN_DISTRICT"]

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, one_hot_features),
        ('num', numerical_pipeline, num_features)
    ])

from sklearn.preprocessing import LabelEncoder
# rebuild training data with encoded y labels

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_cleaned = preprocessor.fit_transform(X_train)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(284, activation='relu', input_shape=(X_cleaned.shape[1],)),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])


    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    history = model.fit(X_cleaned.toarray(), y_train_encoded, epochs=200, validation_split=0.2, batch_size=128)
    
    X_cleaned_test = preprocessor.fit_transform(X_test)
    X_test_dense = X_cleaned_test.toarray()
    test_loss, test_accuracy = model.evaluate(X_test_dense, y_test)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = model.predict(X_test_dense)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)