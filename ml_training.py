# ml_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load dataset
df = pd.read_csv("compensation_data.csv")

# 2. Basic sanity
expected = ['flight_id','airline','delay_minutes','ticket_class','distance_km','region','loyalty_score','base_compensation','final_compensation']
missing = [c for c in expected if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in compensation_data.csv: {missing}")

# 3. Encode categorical columns
le_airline = LabelEncoder()
le_ticket = LabelEncoder()
le_region = LabelEncoder()

df['airline_encoded'] = le_airline.fit_transform(df['airline'])
df['ticket_encoded']  = le_ticket.fit_transform(df['ticket_class'])
df['region_encoded']  = le_region.fit_transform(df['region'])

# 4. Features and target (match order used in app)
X = df[['delay_minutes', 'distance_km', 'loyalty_score',
        'airline_encoded', 'ticket_encoded', 'region_encoded']]
y = df['final_compensation']

# 5. Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 6. Save model and encoders (joblib)
joblib.dump(model, "compensation_model.pkl")
joblib.dump(le_airline, "airline_encoder.pkl")
joblib.dump(le_ticket, "ticket_encoder.pkl")
joblib.dump(le_region, "region_encoder.pkl")

# 7. Optional metrics print
r2 = model.score(X_test, y_test)
print("✅ Training complete. Saved:")
print(" - compensation_model.pkl")
print(" - airline_encoder.pkl")
print(" - ticket_encoder.pkl")
print(" - region_encoder.pkl")
print(f"Model R² on test set: {r2:.3f}")