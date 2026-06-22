import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# Print debug info
print("DEBUG: Script started")
csv_path = os.path.join('data', 'rainfall.csv')
print("DEBUG: CSV path is", csv_path)

df = pd.read_csv(csv_path)
if len(df.columns) == 1:
    print("CSV has only one column, trying with sep=';'")
    df = pd.read_csv(csv_path, sep=';')
print("After re-reading with semicolon, columns:", df.columns)
df.columns = df.columns.str.strip().str.replace('"', '')
print("Cleaned columns:", df.columns)

day_cols = [
    '1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
    '11th','12th','13th','14th','15th','16th','17th','18th','19th','20th',
    '21st','22nd','23rd','24th','25th','26th','27th','28th','29th','30th','31st'
]

print("About to melt data...")
melted = df.melt(id_vars=['state', 'district', 'month'],
                 value_vars=day_cols,
                 var_name='day', value_name='rainfall')
print("After melt,shape:", melted.shape)

melted['rainfall'] = pd.to_numeric(melted['rainfall'], errors='coerce')
melted.dropna(subset=['rainfall'], inplace=True)
imputer = SimpleImputer(strategy="mean")
melted['rainfall'] = imputer.fit_transform(melted[['rainfall']])
print("After melting and cleaning, sample:")
print(melted.head())

print("About to encode categorical variables:")
le_district = LabelEncoder()
le_month = LabelEncoder()
le_day = LabelEncoder()
melted['district_enc'] = le_district.fit_transform(melted['district'])
melted['month_enc'] = le_month.fit_transform(melted['month'].astype(str))
melted['day_enc'] = le_day.fit_transform(melted['day'])
print("Encoded variables.")

print("About to split data/train/test...")
X = melted[['district_enc', 'month_enc', 'day_enc']]
y = melted['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
print("Data shape X_train,X_test:", X_train.shape, X_test.shape)

# ---- Memory Management: Use smaller samples! ----
sample_n = 50000  # Reduce if you still get MemoryError (try 5000 if needed)
test_n = 5000

X_train_samp = X_train[:sample_n]
y_train_samp = y_train[:sample_n]
X_test_samp = X_test[:test_n]
y_test_samp = y_test[:test_n]

results = []


rf_model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=1)
rf_model.fit(X_train_samp, y_train_samp)
rf_pred = rf_model.predict(X_test_samp)
rf_r2 = r2_score(y_test_samp, rf_pred)
rf_mse = mean_squared_error(y_test_samp, rf_pred)
results.append(("Random Forest", rf_r2, rf_mse))
print("Random Forest trained.")


lr = LinearRegression()
lr.fit(X_train_samp, y_train_samp)
lr_pred = lr.predict(X_test_samp)
lr_r2 = r2_score(y_test_samp, lr_pred)
lr_mse = mean_squared_error(y_test_samp, lr_pred)
results.append(("Linear Regression", lr_r2, lr_mse))
print("Linear Regression trained.")

# SVM is very slow on large datasets, so use tiny sample or skip for now
try:
    
    X_train_svm = X_train_samp[:2000]
    y_train_svm = y_train_samp[:2000]
    X_test_svm = X_test_samp[:500]
    y_test_svm = y_test_samp[:500]
    svm = SVR(kernel='rbf')
    svm.fit(X_train_svm, y_train_svm)
    svm_pred = svm.predict(X_test_svm)
    svm_r2 = r2_score(y_test_svm, svm_pred)
    svm_mse = mean_squared_error(y_test_svm, svm_pred)
    results.append(("SVM", svm_r2, svm_mse))
    print("SVM trained.")
except Exception as e:
    print("SVM train failed (likely due to memory/slow):", e)

print("| Model             | R² Score |   MSE   |")
print("|-------------------|----------|---------|")
for name, r2, mse in results:
    print(f"| {name:<17} | {r2:8.4f} | {mse:7.4f} |")

model_names = [name for name, _, _ in results]
r2_scores = [r2 for _, r2, _ in results]
plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores, color=['orange', 'limegreen', 'skyblue'][:len(results)],width=0.4)
plt.title("Model R² Score Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.show()