import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===== טעינת הדאטה מכל הנבדקים =====

subjects = [f'S{i}' for i in range(2, 18)]  # S2 עד S17
base_path = '/Users/galshemesh/Downloads/WESAD/'

all_data = []

for subject in subjects:
    file_path = f'{base_path}{subject}/{subject}.pkl'
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')

        signals = data['signal']
        labels = data['label']

        nonzero_idx = np.where(labels != 0)[0][0]

        eda = signals['chest']['EDA'].flatten()[nonzero_idx:]
        temp = signals['chest']['Temp'].flatten()[nonzero_idx:]
        label = labels[nonzero_idx:]

        min_len = min(len(eda), len(temp), len(label))
        eda = eda[:min_len]
        temp = temp[:min_len]
        label = label[:min_len]

        df_subject = pd.DataFrame({

            'eda': eda,
            'temp': temp,
            'label': label,
            'subject': subject
        })

        df_subject = df_subject[df_subject['label'].isin([1, 2, 3])]
        df_subject['is_stress'] = df_subject['label'].apply(lambda x: 1 if x == 2 else 0)

        all_data.append(df_subject)

    except FileNotFoundError:
        print(f"⚠️ קובץ חסר: {subject} - מדלגים עליו...")

# איחוד כל הדאטה
df_all = pd.concat(all_data, ignore_index=True)

print("\nדאטה משולב מוכן:")
print(df_all.head())

print("\nהתפלגות סטרס:")
print(df_all['is_stress'].value_counts())

print("\nרשימת נבדקים:")
print(df_all['subject'].unique())

# ===== הוספת פיצ'רים חדשים (Feature Engineering) =====

df_all['delta_eda'] = df_all['eda'].diff().fillna(0)
df_all['delta_temp'] = df_all['temp'].diff().fillna(0)

df_all['eda_mean_5'] = df_all['eda'].rolling(window=5, min_periods=1).mean()
df_all['temp_mean_5'] = df_all['temp'].rolling(window=5, min_periods=1).mean()

df_all['eda_std_5'] = df_all['eda'].rolling(window=5, min_periods=1).std().fillna(0)
df_all['temp_std_5'] = df_all['temp'].rolling(window=5, min_periods=1).std().fillna(0)

# ===== חלוקה לאימון וטסט =====

train_df = df_all[df_all['subject'] != 'S9']
test_df = df_all[df_all['subject'] == 'S9']

features = ['eda', 'temp', 'delta_eda', 'delta_temp', 'eda_mean_5', 'temp_mean_5', 'eda_std_5', 'temp_std_5']

X_train = train_df[features]
y_train = train_df['is_stress']

X_test = test_df[features]
y_test = test_df['is_stress']

# ===== נורמליזציה =====

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== בניית מודל XGBoost =====

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# ===== אימון המודל עם מעקב =====

xgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=True
)

# ===== שמירת תוצאות האימון =====

results = xgb_model.evals_result()

# ===== ציור Learning Curve =====

epochs = len(results['validation_0']['logloss'])
x_axis = range(epochs)

plt.figure()
plt.plot(x_axis, results['validation_0']['logloss'], label='Train Loss')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# ===== חיזוי על נבדק S9 =====

y_pred = xgb_model.predict(X_test_scaled)

print("\nתוצאות על נבדק S9:")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== שמירת חיזויים =====
predictions_df = pd.DataFrame({'Real': y_test.values, 'Predicted': y_pred})
predictions_df.to_csv('predictions_S9.csv', index=False)
print("\nקובץ 'predictions_S9.csv' נשמר בהצלחה.")

# ===== שמירת מודל מאומן =====
joblib.dump(xgb_model, 'xgboost_model.pkl')
print("המודל נשמר כ-'xgboost_model.pkl'.")
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== טעינת הדאטה מכל הנבדקים =====

# כל הנבדקים מ-S2 עד S17
subjects = [f'S{i}' for i in range(2, 18)]

base_path = '/Users/galshemesh/Downloads/WESAD/'

all_data = []

for subject in subjects:
    file_path = f'{base_path}{subject}/{subject}.pkl'
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')

        signals = data['signal']
        labels = data['label']

        nonzero_idx = np.where(labels != 0)[0][0]

        eda = signals['chest']['EDA'].flatten()[nonzero_idx:]
        temp = signals['chest']['Temp'].flatten()[nonzero_idx:]
        label = labels[nonzero_idx:]

        min_len = min(len(eda), len(temp), len(label))
        eda = eda[:min_len]
        temp = temp[:min_len]
        label = label[:min_len]

        df_subject = pd.DataFrame({
            'eda': eda,
            'temp': temp,
            'label': label,
            'subject': subject
        })

        df_subject = df_subject[df_subject['label'].isin([1, 2, 3])]
        df_subject['is_stress'] = df_subject['label'].apply(lambda x: 1 if x == 2 else 0)

        all_data.append(df_subject)

    except FileNotFoundError:
        print(f"⚠️ קובץ חסר: {subject} - מדלגים עליו...")

# איחוד כל הדאטה
df_all = pd.concat(all_data, ignore_index=True)

print("\n📊 דאטה משולב מוכן:")
print(df_all.head())
print("\n🧮 התפלגות סטרס:")
print(df_all['is_stress'].value_counts())
print("\n🧮 רשימת נבדקים:")
print(df_all['subject'].unique())

# ===== הוספת פיצ'רים חדשים (Feature Engineering) =====

df_all['delta_eda'] = df_all['eda'].diff().fillna(0)
df_all['delta_temp'] = df_all['temp'].diff().fillna(0)

df_all['eda_mean_5'] = df_all['eda'].rolling(window=5, min_periods=1).mean()
df_all['temp_mean_5'] = df_all['temp'].rolling(window=5, min_periods=1).mean()

df_all['eda_std_5'] = df_all['eda'].rolling(window=5, min_periods=1).std().fillna(0)
df_all['temp_std_5'] = df_all['temp'].rolling(window=5, min_periods=1).std().fillna(0)

# ===== חלוקה לאימון וטסט =====

# אימון על כל הנבדקים חוץ מ-S9
train_df = df_all[df_all['subject'] != 'S9']
test_df = df_all[df_all['subject'] == 'S9']

# רשימת פיצ'רים לשימוש
features = ['eda', 'temp', 'delta_eda', 'delta_temp', 'eda_mean_5', 'temp_mean_5', 'eda_std_5', 'temp_std_5']

X_train = train_df[features]
y_train = train_df['is_stress']

X_test = test_df[features]
y_test = test_df['is_stress']

# ===== נורמליזציה =====

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== בניית מודל XGBoost =====

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# אימון המודל
xgb_model.fit(X_train_scaled, y_train)

# ===== חיזוי =====

y_pred = xgb_model.predict(X_test_scaled)

# ===== תוצאות =====

print("\n📊 Confusion Matrix (בדיקה על S9):")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report (בדיקה על S9):")
print(classification_report(y_test, y_pred))
