from flask import Flask, render_template, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import matplotlib
from BDS import df  # Importar el DataFrame desde BDS.py
from flask import send_file
import plotly.graph_objects as go



# Configurar Matplotlib para usar un backend no interactivo
matplotlib.use('Agg')

app = Flask(__name__)

# Escalar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['ATSM'].values.reshape(-1, 1))
# Ruta para descargar datos
@app.route('/download-data')
def download_data():
    # Guardar el DataFrame en un archivo CSV
    csv_file_path = 'datos_atms.csv'  # Puedes cambiar el nombre del archivo como desees
    df.to_csv(csv_file_path, index=False)

    # Enviar el archivo CSV como respuesta para su descarga
    return send_file(csv_file_path, as_attachment=True)
# Preparar datos
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



@app.route('/analisis')
def analisis():
    # Crear gráfica interactiva usando Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['ATSM'], mode='lines+markers', name='Histórico'))

    fig.update_layout(
    title={
        'text': 'Datos de ATSM desde 2014-2023',
        'x': 0.5,  # Centra el título
        'xanchor': 'center',  # Asegura que el anclaje esté en el centro
        'yanchor': 'top'
    },
    xaxis_title='Fecha',
    yaxis_title='ATSM',
    hovermode='closest'
)


    # Generar el HTML para la gráfica
    graph_html = fig.to_html(full_html=False)

    return render_template('analisis.html', graph_html=graph_html)


@app.route('/proyeccion')
def proyeccion():
    return render_template('proyeccion.html')

@app.route('/contacto')
def contacto():
    return render_template('contacto.html')


# Modelo LSTM
n_steps = 12  # Pasos de tiempo para la LSTM
X, y = prepare_data(scaled_data, n_steps)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(150, activation='tanh', return_sequences=True, input_shape=(n_steps, 1)),
    Dropout(0.2),
    LSTM(100, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)

# Ruta principal
@app.route('/')
def home():
    # Crear gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(df['Fecha'], df['ATSM'], label='Histórico', color='blue')
    plt.xlabel('Fecha')
    plt.ylabel('ATSM')
    plt.title('Datos de ATSM desde 2014-2023')
    plt.legend()

    # Convertir gráfica a imagen base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Cerrar la figura para liberar memoria
    return render_template('index.html', plot_url=plot_url)

# Ruta para la proyección
@app.route('/forecast')
def forecast():
    # Proyección a 10 años
    last_sequence = scaled_data[-n_steps:]
    forecast = []
    for _ in range(120):
        predicted_value = model.predict(last_sequence.reshape(1, n_steps, 1), verbose=0)[0, 0]
        forecast.append(predicted_value)
        
        # Aumentar la variación en las predicciones
        noise = np.random.normal(0, 0.3)
        last_sequence = np.append(last_sequence[1:], predicted_value + noise).reshape(-1, 1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Generar nuevas fechas para la proyección
    new_dates = pd.date_range(start=df['Fecha'].iloc[-1] + pd.DateOffset(months=1), periods=120, freq='MS')

    # Crear gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(df['Fecha'], df['ATSM'], label='Histórico', color='blue')
    plt.plot(new_dates, forecast, label='Proyección', color='red')
    plt.xlabel('Fecha')
    plt.ylabel('ATSM')
    plt.title('Proyección de ATSM a 10 años')
    plt.legend()

    # Ajustar los límites del eje Y
    min_y = min(np.min(df['ATSM']), np.min(forecast))
    max_y = max(np.max(df['ATSM']), np.max(forecast))
    plt.ylim(min_y - 0.1, max_y + 0.1)

    # Convertir gráfica a imagen base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()  # Cerrar la figura para liberar memoria
    return jsonify({'forecast_plot': plot_url})

# Ruta para descargar datos
@app.route('/download')
def download():
    # Convertir DataFrame a CSV
    csv = df.to_csv(index=False)
    # Convertir CSV a bytes
    response = app.response_class(
        response=csv,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=datos_atm.csv"}
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
