




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri yükleme
data = pd.read_csv('01_Data/train_and_test_2.csv')

# Kategorik sütunları kodlamak
label_encoder = LabelEncoder()
data['booker_country'] = label_encoder.fit_transform(data['booker_country'])
data['hotel_country'] = label_encoder.fit_transform(data['hotel_country'])

# Tarih değerlerini datetime nesnelerine dönüştürme
data['checkin'] = pd.to_datetime(data['checkin'])
data['checkout'] = pd.to_datetime(data['checkout'])

# Tarihler arasındaki farkı hesaplama
data['stay_duration'] = (data['checkout'] - data['checkin']).dt.days

# Eğitim, test ve doğrulama veri setlerini ayırma
train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.2)

# Bağımsız değişkenler ve hedef değişken
# "device_class" sütununu One-Hot Encoding ile dönüştürme


X_train = train[['user_id', 'stay_duration', 'device_class', 'affiliate_id', 'booker_country', 'hotel_country', 'dcount', 'icount']]
y_train = train['city_id']

X_val = validation[['user_id', 'stay_duration', 'device_class', 'affiliate_id', 'booker_country', 'hotel_country', 'dcount', 'icount']]
y_val = validation['city_id']
X_train = pd.get_dummies(X_train, columns=['device_class'], drop_first=True)
X_val = pd.get_dummies(X_val, columns=['device_class'], drop_first=True)
# Lojistik regresyon modelini eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Lojistik Regresyon Doğruluk: {accuracy}")
