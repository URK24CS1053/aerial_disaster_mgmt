import streamlit as st

# Title of the Dashboard
st.title('SAR System Dashboard')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Example feature: Input for Search and Rescue Operation
operation_type = st.sidebar.selectbox('Select Operation Type', ['Search', 'Rescue'])

# Display input feature
if operation_type == 'Search':
    location = st.sidebar.text_input('Enter Location for Search')
    st.write('Searching in:', location)
elif operation_type == 'Rescue':
    victim_count = st.sidebar.number_input('Number of Victims to Rescue', min_value=1)
    st.write('Preparing rescue for', victim_count, 'victims')

# Main panel
st.header('Operation Status')

# Dummy data for current operations
st.write('Current Operations:')

# Example table for displaying operations
import pandas as pd

data = { 'Operation ID': [1, 2], 'Location': ['Location A', 'Location B'], 'Status': ['Ongoing', 'Completed'] }

operations_df = pd.DataFrame(data)
st.dataframe(operations_df)

# Footer
st.sidebar.markdown('> Built with Streamlit')
st.markdown('### SAR System Dashboard is developed to aid in Search and Rescue operations.')
