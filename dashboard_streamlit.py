import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the Dashboard
st.title('Aerial Disaster Management Dashboard')

# System Metrics Page
st.header('System Metrics')
# Example metric data
metrics_data = pd.DataFrame({
    'Metric': ['CPU Usage', 'Memory Usage', 'Disk Space'],
    'Value': [70, 60, 80]
})
fig_metrics = px.bar(metrics_data, x='Metric', y='Value', title='System Metrics')
st.plotly_chart(fig_metrics)

# Operating Modes Page
st.header('Operating Modes')
operating_modes = ['Normal', 'Emergency', 'Maintenance']
selected_mode = st.selectbox('Select Operating Mode', operating_modes)
st.write(f'You selected: {selected_mode}')

# Test Scenarios Page
st.header('Test Scenarios')
test_scenarios = pd.DataFrame({
    'Scenario': ['Fire', 'Flood', 'Earthquake'],
    'Status': ['Completed', 'In Progress', 'Pending']
})
fig_scenarios = px.pie(test_scenarios, names='Scenario', values='Status', title='Test Scenarios Status')
st.plotly_chart(fig_scenarios)

# Feature Vectors Page
st.header('Feature Vectors')
feature_vectors = pd.DataFrame({
    'Feature': ['Temperature', 'Humidity', 'Wind Speed'],
    'Importance': [0.8, 0.6, 0.5]
})
fig_features = px.bar(feature_vectors, x='Feature', y='Importance', title='Feature Importances')
st.plotly_chart(fig_features)

# Architecture Page
st.header('Architecture')
st.write('Architecture diagram will be shown here.')  # Placeholder for architecture diagram

# Data Flow Page
st.header('Data Flow')
st.write('Data flow diagram will be shown here.')  # Placeholder for data flow diagram

# Quick Start Page
st.header('Quick Start')
st.write('Instructions on how to use the system.')

# ML Model Details Page
st.header('ML Model Details')
st.write('Details about the machine learning model will be displayed here.')

# Troubleshooting Page
st.header('Troubleshooting')
st.write('Common issues and troubleshooting steps will be listed here.')

# Project Structure Page
st.header('Project Structure')
st.write('Project structure details will be displayed here.')