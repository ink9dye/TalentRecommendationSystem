import streamlit as st
from src.interface.dashboard import show_main_dashboard
from src.infrastructure.mock_data import mock_talents, mock_graph # 假设你已迁入

st.set_page_config(layout="wide")
show_main_dashboard(mock_talents, mock_graph)