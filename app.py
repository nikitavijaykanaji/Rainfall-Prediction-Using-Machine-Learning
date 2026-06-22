import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

csv_path = os.path.join('data', 'rainfall.csv')


df = pd.read_csv(csv_path)
if len(df.columns) == 1:
    df = pd.read_csv(csv_path, sep=';')
df.columns = df.columns.str.strip().str.replace('"', '')
print("ACTUAL COLUMNS:", df.columns.tolist())


import re
day_cols = [col for col in df.columns if re.fullmatch(r'\d+(st|nd|rd|th)', col)]

print("Detected day columns:", day_cols)

if not day_cols:
    raise RuntimeError("No day columns detected! Please check your CSV and delimiter!")

melted = df.melt(id_vars=['state', 'district', 'month'], value_vars=day_cols,
                 var_name='day', value_name='rainfall')

melted['rainfall'] = pd.to_numeric(melted['rainfall'], errors='coerce')
melted.dropna(subset=['rainfall'], inplace=True)
imputer = SimpleImputer(strategy="mean")
melted['rainfall'] = imputer.fit_transform(melted[['rainfall']])

le_district = LabelEncoder()
le_month = LabelEncoder()
le_day = LabelEncoder()

melted['district_enc'] = le_district.fit_transform(melted['district'])
melted['month_enc'] = le_month.fit_transform(melted['month'].astype(str))
melted['day_enc'] = le_day.fit_transform(melted['day'])

X = melted[['district_enc', 'month_enc', 'day_enc']]
y = melted['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)

print("Random Forest R2:", model.score(X_test, y_test))


districts = sorted(melted['district'].unique())
months = sorted(melted['month'].astype(int).unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected = {"district": "", "month": "", "day": ""}
    days = [int(x[:-2]) for x in day_cols]
    if request.method == 'POST':
        district = request.form['district']
        month = request.form['month']
        day = request.form['day']
        selected = {"district": district, "month": month, "day": day}
        try:
            district_enc = le_district.transform([district])[0]
            month_enc = le_month.transform([month])[0]
            
            for d in day_cols:
                if d.startswith(day) and d == f"{day}{d[-2:]}":
                    day_str = d
                    break
            else:
                day_str = f"{day}th"
            day_enc = le_day.transform([day_str])[0]
            pred = model.predict([[district_enc, month_enc, day_enc]])[0]
            result = f"Predicted Rainfall : {pred:.2f}mm"
        except Exception as e:
            result = f"Invalid input or model error: {str(e)}"
    return render_template('index.html', districts=districts, months=months, days=days, result=result, selected=selected)

if __name__== '__main__':
    app.run(debug=True)