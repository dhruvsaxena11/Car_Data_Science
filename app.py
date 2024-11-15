import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from transformers import pipeline
from PIL import Image
import numpy as np

# Function to simulate sensor data
def generate_sensor_data(num_entries):
    data = []
    timestamp = datetime.now()
    for _ in range(num_entries):
        entry = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "engine_temp": random.randint(80, 110),
            "oil_pressure": random.randint(15, 50),
            "tire_pressure": random.randint(29, 35),
            "battery_voltage": round(random.uniform(11.0, 13.0), 1)
        }
        # Introduce anomalies
        if random.random() < 0.1:
            entry["engine_temp"] = random.randint(105, 120)
        if random.random() < 0.1:
            entry["oil_pressure"] = random.randint(10, 20)
        if random.random() < 0.05:
            entry["battery_voltage"] = round(random.uniform(10.5, 11.5), 1)
        data.append(entry)
        timestamp += timedelta(minutes=5)
    return pd.DataFrame(data)

# Generate sensor data
sensor_data = generate_sensor_data(10)

# Plot sensor data
def plot_sensor_data(df):
    timestamps = pd.to_datetime(df["timestamp"])
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensor Data', fontsize=16)

    thresholds = {"engine_temp": 100, "oil_pressure": 25, "tire_pressure": 28, "battery_voltage": 11.5}

    axs[0, 0].plot(timestamps, df["engine_temp"], color='red')
    axs[0, 0].axhline(y=thresholds["engine_temp"], color='green', linestyle='--')
    axs[0, 0].set_title("Engine Temperature (°C)")

    axs[0, 1].plot(timestamps, df["oil_pressure"], color='blue')
    axs[0, 1].axhline(y=thresholds["oil_pressure"], color='orange', linestyle='--')
    axs[0, 1].set_title("Oil Pressure (psi)")

    axs[1, 0].plot(timestamps, df["tire_pressure"], color='green')
    axs[1, 0].axhline(y=thresholds["tire_pressure"], color='purple', linestyle='--')
    axs[1, 0].set_title("Tire Pressure (psi)")

    axs[1, 1].plot(timestamps, df["battery_voltage"], color='black')
    axs[1, 1].axhline(y=thresholds["battery_voltage"], color='brown', linestyle='--')
    axs[1, 1].set_title("Battery Voltage (V)")

    plt.tight_layout()
    fig_path = "sensor_plot.png"
    plt.savefig(fig_path)
    plt.close(fig)
    return Image.open(fig_path)

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Generate recommendations and detect anomalies without model
def analyze_data(image, df):
    try:
        # Ensure image is a PIL object
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Damage analysis
        damage_output = pipe(inputs=image)[0]['generated_text'] if image else "No image uploaded."

        # Anomaly detection based on sensor thresholds
        anomalies = []
        thresholds = {"engine_temp": 100, "oil_pressure": 25, "tire_pressure": 28, "battery_voltage": 11.5}
        
        if any(df["engine_temp"] > thresholds["engine_temp"]):
            anomalies.append("Engine temperature is above normal detected.")
        if any(df["oil_pressure"] < thresholds["oil_pressure"]):
            anomalies.append("Oil pressure is below normal detected.")
        if any(df["tire_pressure"] < thresholds["tire_pressure"]):
            anomalies.append("Tire pressure is below normal detected.")
        if any(df["battery_voltage"] < thresholds["battery_voltage"]):
            anomalies.append("Battery voltage is below normal detected.")

        # Recommendations
        recommendations = "Analysis:\n\n"
        recommendations += "\n".join([f"- {anomaly}" for anomaly in anomalies]) + "\n\n"
        recommendations += f"Damage Analysis:\n- {damage_output}\n\n"
        recommendations += "Recommendations:\n\n- Please make repairs at a nearby service station for optimal car performance."

        return recommendations, plot_sensor_data(df)

    except Exception as e:
        return f"Error occurred: {str(e)}", None

# Gradio UI
with gr.Blocks(css=".output-text { font-family: 'Arial'; color: #222; font-size: 1rem; }") as app:
    gr.Markdown("# 🚗 Car Health Report Generation using Generative AI")
    with gr.Row():
        car_image = gr.Image(type="pil", label="Upload Car Part Damage Image")
    with gr.Row():
        display_graph = gr.Image(value=plot_sensor_data(sensor_data), type="pil", label="Sensor Data Over Time")
        recommendations = gr.Textbox(label="Analysis & Recommendations", placeholder="Insights will appear here...")
    data_table = gr.Dataframe(value=sensor_data, label="Generated Sensor Data (Table View)", row_count=(10, "fixed"), interactive=False)

    # Set up Gradio interaction with wrapped DataFrame component
    car_image.change(fn=analyze_data, inputs=[car_image, data_table], outputs=[recommendations, display_graph])

app.launch()
