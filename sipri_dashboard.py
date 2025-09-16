"""
SIPRI Arms Transfer Dashboard
A comprehensive Streamlit dashboard for analyzing SIPRI arms transfer data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="SIPRI Arms Transfer Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - CHANGED GREY TO BLACK BACKGROUNDS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #000000;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    div[data-testid="metric-container"] label {
        color: #cccccc !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    }
    .info-box {
        background-color: #1e3d59;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Color palette
COLORS = {
    'supplier': '#1f77b4',  # Blue
    'recipient': '#ff7f0e',  # Orange
    'positive': '#2ca02c',   # Green
    'negative': '#d62728',   # Red
    'neutral': '#9467bd',    # Purple
    'background': '#000000'   # Changed to black
}

# ==================== DATA LOADING FUNCTIONS ====================

@st.cache_data(show_spinner=False)
def load_sipri_data(file_path="SIPRI_Arms_transfers.csv", uploaded_file=None):
    """
    Load SIPRI data from CSV file with error handling for various formats
    """
    try:
        # Determine source - prioritize local file
        if os.path.exists(file_path) and uploaded_file is None:
            source = file_path
            st.info(f"Loading local file: {file_path}")
        elif uploaded_file is not None:
            source = uploaded_file
            st.info("Loading uploaded file")
        else:
            st.error(f"File '{file_path}' not found. Please ensure the CSV file is in the same directory as this script.")
            return None
        
        # Read the CSV, skipping the header rows with metadata
        for skip_rows in [10, 11, 9, 0]:  # SIPRI files often have metadata in first rows
            try:
                df = pd.read_csv(source, skiprows=skip_rows, encoding='utf-8', low_memory=False)
                
                # Check if we have valid data (should have typical SIPRI columns)
                if len(df.columns) > 10 and len(df) > 0:
                    # Check for key SIPRI columns
                    key_cols = ['Supplier', 'Recipient', 'Designation', 'Description']
                    if any(col in df.columns for col in key_cols):
                        break
            except:
                continue
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Standardize column names
        column_mapping = {
            'Numbers delivered': 'Numbers delivered',
            'Numbers delivered is estimate': 'Numbers delivered is estimate',
            'Delivery year': 'Delivery year',
            'Delivery year is estimate': 'Delivery year is estimate',
            'TIV delivery values': 'TIV delivery values',
            'TIV deal unit': 'TIV deal unit',
            'SIPRI AT Database ID': 'Database ID',
            'Armament category': 'Armament category',
            'Order date': 'Order date',
            'Order date is estimate': 'Order date is estimate',
            'SIPRI estimate': 'SIPRI estimate',
            'Local production': 'Local production'
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Convert numeric columns with improved error handling
        numeric_cols = ['Numbers delivered', 'Delivery year', 'Order date', 
                       'TIV delivery values', 'TIV deal unit', 'Database ID']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Fill missing values
        if 'TIV delivery values' in df.columns:
            df['TIV delivery values'] = df['TIV delivery values'].fillna(0)
        
        # Add helpful derived columns
        if 'Delivery year' in df.columns:
            # Clean up year data - remove invalid years
            df = df[df['Delivery year'].notna()]
            df = df[(df['Delivery year'] >= 1900) & (df['Delivery year'] <= 2100)]
            df['Decade'] = (df['Delivery year'] // 10) * 10
            df['Decade'] = df['Decade'].apply(lambda x: f"{int(x)}s" if pd.notna(x) else "Unknown")
            
        if 'Order date' in df.columns and 'Delivery year' in df.columns:
            df['Delivery delay (years)'] = df['Delivery year'] - df['Order date']
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check that your CSV file is properly formatted SIPRI data.")
        return None

@st.cache_data
def process_data(df):
    """Process and enhance the dataframe with additional useful columns"""
    if df is None:
        return None
    
    try:
        # Create simplified weapon categories
        if 'Description' in df.columns:
            # Extract main weapon type
            weapon_categories = {
                'aircraft': ['aircraft', 'helicopter', 'UAV', 'plane', 'drone'],
                'missiles': ['missile', 'SAM', 'torpedo', 'bomb', 'rocket'],
                'ships': ['ship', 'boat', 'submarine', 'frigate', 'destroyer', 'carrier', 'corvette'],
                'vehicles': ['tank', 'armoured', 'APC', 'IFV', 'vehicle', 'personnel carrier', 'truck'],
                'artillery': ['gun', 'howitzer', 'mortar', 'launcher', 'artillery', 'cannon'],
                'air_defence': ['air-defence', 'anti-aircraft', 'SAM system', 'CIWS'],
                'sensors': ['radar', 'sonar', 'sensor', 'EW', 'electronic'],
                'engines': ['engine', 'turbine', 'turbofan', 'turbojet', 'diesel']
            }
            
            def categorize_weapon(desc):
                if pd.isna(desc):
                    return 'Other'
                desc_lower = str(desc).lower()
                for category, keywords in weapon_categories.items():
                    if any(keyword in desc_lower for keyword in keywords):
                        return category.replace('_', ' ').title()
                return 'Other'
            
            df['Weapon Category'] = df['Description'].apply(categorize_weapon)
        
        # Ensure data integrity
        if 'Delivery year' in df.columns:
            df = df[df['Delivery year'].notna()]
            
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return df

# ==================== INFORMATION PANELS ====================

def show_sipri_glossary():
    """Display comprehensive SIPRI terminology and explanations"""
    st.markdown("""
    <div class="info-box">
    <h3>📚 SIPRI Data Guide & Glossary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Key Terms Explained
        
        **TIV (Trend-Indicator Value)**
        - SIPRI's unique unit to measure arms transfer volumes
        - Represents military capability, NOT financial value
        - Based on production costs of core units, adjusted for quality
        - Allows comparison across different weapon types and time periods
        
        **Major Conventional Weapons**
        - Aircraft: Combat planes, transport aircraft, trainers
        - Helicopters: Attack, transport, ASW helicopters  
        - Ships: Warships over 100 tonnes
        - Armoured vehicles: Tanks, APCs, IFVs
        - Artillery: Large caliber systems (>100mm)
        - Missiles: Guided weapons and torpedoes
        - Air defence: SAM systems, radars
        
        **Transfer Status Types**
        - New: Newly produced equipment
        - Second hand: Previously used equipment
        - Second hand but modernized: Used but upgraded
        - Licensed production: Built locally under license
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data Fields Explained
        
        **Supplier/Recipient**: Countries or entities involved in transfer
        
        **Designation**: Specific model/type (e.g., F-16, T-72)
        
        **Description**: General category (e.g., fighter aircraft, tank)
        
        **Order date**: When deal was agreed (may be estimate)
        
        **Delivery year**: When weapons were actually transferred
        
        **Numbers delivered**: Quantity transferred in that year
        
        **Status**: New, second-hand, or modernized
        
        **SIPRI estimate**: Whether SIPRI estimated missing data
        
        **Local production**: If produced under license in recipient country
        
        ### 💡 How to Interpret
        - Higher TIV = Greater military capability transferred
        - Delivery delays common in arms trade (check Order vs Delivery dates)
        - Second-hand doesn't mean inferior (often modernized)
        - Local production indicates technology transfer
        """)

# ==================== VISUALIZATION FUNCTIONS ====================

def create_top_suppliers_chart(df, n=10):
    """Create bar chart of top suppliers by TIV"""
    if 'Supplier' not in df.columns or 'TIV delivery values' not in df.columns:
        return None
    
    top_suppliers = df.groupby('Supplier')['TIV delivery values'].sum().nlargest(n).reset_index()
    
    fig = px.bar(
        top_suppliers, 
        x='TIV delivery values', 
        y='Supplier',
        orientation='h',
        title=f'Top {n} Arms Suppliers by Total TIV',
        labels={'TIV delivery values': 'Total TIV (Trend Indicator Value)', 'Supplier': ''},
        color='TIV delivery values',
        color_continuous_scale='Blues',
        text='TIV delivery values'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_top_recipients_chart(df, n=10):
    """Create bar chart of top recipients by TIV"""
    if 'Recipient' not in df.columns or 'TIV delivery values' not in df.columns:
        return None
    
    top_recipients = df.groupby('Recipient')['TIV delivery values'].sum().nlargest(n).reset_index()
    
    fig = px.bar(
        top_recipients, 
        x='TIV delivery values', 
        y='Recipient',
        orientation='h',
        title=f'Top {n} Arms Recipients by Total TIV',
        labels={'TIV delivery values': 'Total TIV (Trend Indicator Value)', 'Recipient': ''},
        color='TIV delivery values',
        color_continuous_scale='Oranges',
        text='TIV delivery values'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_time_series_chart(df):
    """Create time series of TIV by year"""
    if 'Delivery year' not in df.columns or 'TIV delivery values' not in df.columns:
        return None
    
    yearly_data = df.groupby('Delivery year')['TIV delivery values'].sum().reset_index()
    yearly_data = yearly_data.sort_values('Delivery year')
    
    # Remove any years that are NaN or unrealistic
    yearly_data = yearly_data[yearly_data['Delivery year'].notna()]
    yearly_data = yearly_data[(yearly_data['Delivery year'] >= 1950) & (yearly_data['Delivery year'] <= 2030)]
    
    fig = px.line(
        yearly_data,
        x='Delivery year',
        y='TIV delivery values',
        title='Global Arms Transfers Over Time (TIV)',
        labels={'TIV delivery values': 'Total TIV', 'Delivery year': 'Year'},
        markers=True
    )
    fig.update_traces(line_color=COLORS['supplier'], line_width=3)
    fig.update_layout(height=400, hovermode='x unified')
    
    # Add annotation for what TIV means
    fig.add_annotation(
        text="TIV = Trend Indicator Value (military capability, not $)",
        xref="paper", yref="paper",
        x=0, y=1.1, showarrow=False,
        font=dict(size=10, color="gray")
    )
    return fig

def create_weapon_category_pie(df):
    """Create pie chart of weapon categories"""
    if 'Weapon Category' not in df.columns or 'TIV delivery values' not in df.columns:
        return None
    
    category_data = df.groupby('Weapon Category')['TIV delivery values'].sum().reset_index()
    category_data = category_data.nlargest(8, 'TIV delivery values')
    
    fig = px.pie(
        category_data,
        values='TIV delivery values',
        names='Weapon Category',
        title='Weapon Categories by Total TIV',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def create_bilateral_heatmap(df, top_n=15):
    """Create heatmap of bilateral trade flows"""
    if not all(col in df.columns for col in ['Supplier', 'Recipient', 'TIV delivery values']):
        return None
    
    # Get top suppliers and recipients
    top_suppliers = df.groupby('Supplier')['TIV delivery values'].sum().nlargest(top_n).index
    top_recipients = df.groupby('Recipient')['TIV delivery values'].sum().nlargest(top_n).index
    
    # Filter data
    filtered_df = df[df['Supplier'].isin(top_suppliers) & df['Recipient'].isin(top_recipients)]
    
    if len(filtered_df) == 0:
        return None
    
    # Create pivot table
    pivot = filtered_df.pivot_table(
        values='TIV delivery values',
        index='Recipient',
        columns='Supplier',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Supplier", y="Recipient", color="Total TIV"),
        title=f"Arms Transfer Matrix: Top {top_n} Countries (Total TIV)",
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig.update_layout(height=600)
    return fig

# ==================== NEW FUNCTION: TOP RECIPIENTS & SUPPLIERS COMPARISON ====================

def create_top_countries_comparison(df, n=10):
    """Create side-by-side comparison of top suppliers and recipients"""
    if not all(col in df.columns for col in ['Supplier', 'Recipient', 'TIV delivery values']):
        return None
    
    # Get top suppliers
    top_suppliers = df.groupby('Supplier')['TIV delivery values'].sum().nlargest(n).reset_index()
    top_suppliers['Type'] = 'Supplier'
    top_suppliers.rename(columns={'Supplier': 'Country'}, inplace=True)
    
    # Get top recipients
    top_recipients = df.groupby('Recipient')['TIV delivery values'].sum().nlargest(n).reset_index()
    top_recipients['Type'] = 'Recipient'
    top_recipients.rename(columns={'Recipient': 'Country'}, inplace=True)
    
    # Combine data
    combined = pd.concat([top_suppliers, top_recipients])
    
    # Create grouped bar chart
    fig = px.bar(
        combined,
        x='Country',
        y='TIV delivery values',
        color='Type',
        title=f'Top {n} Arms Suppliers vs Recipients by Total TIV',
        labels={'TIV delivery values': 'Total TIV'},
        color_discrete_map={'Supplier': COLORS['supplier'], 'Recipient': COLORS['recipient']},
        barmode='group'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    # Title and Introduction
    st.title("🚀 SIPRI Arms Transfer Dashboard")
    st.markdown("**Analyzing global arms transfers using SIPRI (Stockholm International Peace Research Institute) data**")
    
    # Info box about SIPRI
    st.markdown("""
    <div class="info-box">
    <b>About SIPRI:</b> The Stockholm International Peace Research Institute is an independent international institute 
    dedicated to research into conflict, armaments, arms control and disarmament. Their Arms Transfer Database is the 
    most comprehensive public source of information on international arms transfers.
    </div>
    """, unsafe_allow_html=True)
    
    # Terminology Expander
    with st.expander("📖 **Understanding SIPRI Data - Click to Learn More**", expanded=False):
        show_sipri_glossary()
    
    st.markdown("---")
    
    # Data Loading Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Source")
        
        # Try to load local file first
        df = load_sipri_data()
        
        # Option to upload different file
        st.markdown("**Or upload a different SIPRI CSV file:**")
        uploaded_file = st.file_uploader(
            "Choose a SIPRI CSV file",
            type=['csv'],
            help="Upload a different SIPRI arms transfer data CSV file if needed"
        )
        
        if uploaded_file is not None:
            df = load_sipri_data(uploaded_file=uploaded_file)
    
    with col2:
        if df is not None:
            st.success(f"✅ Data loaded successfully!")
            st.metric("Total Records", f"{len(df):,}")
            if 'Delivery year' in df.columns:
                year_min = df['Delivery year'].min()
                year_max = df['Delivery year'].max()
                st.metric("Date Range", f"{year_min:.0f} - {year_max:.0f}")
    
    if df is None:
        st.error("""
        ⚠️ **No data loaded!**
        
        Please ensure 'SIPRI_Arms_transfers.csv' is in the same directory as this script,
        or upload a SIPRI CSV file using the uploader above.
        
        You can download SIPRI data from: https://www.sipri.org/databases/armstransfers
        """)
        st.stop()
    
    # Process data
    df = process_data(df)
    
    # ==================== SIDEBAR FILTERS ====================
    
    st.sidebar.header("🔍 Filters")
    st.sidebar.markdown("*Adjust filters to explore specific aspects of the data*")
    
    # Year filter with improved handling
    if 'Delivery year' in df.columns:
        year_min = int(df['Delivery year'].min()) if not df['Delivery year'].isna().all() else 1950
        year_max = int(df['Delivery year'].max()) if not df['Delivery year'].isna().all() else 2024
        
        # Ensure valid range
        if year_min > year_max:
            year_min, year_max = 1950, 2024
        
        year_range = st.sidebar.slider(
            "📅 Delivery Year Range",
            min_value=year_min,
            max_value=year_max,
            value=(max(year_min, 2000), year_max),
            help="Filter data by delivery year"
        )
    else:
        year_range = None
    
    # Supplier filter
    if 'Supplier' in df.columns:
        all_suppliers = sorted(df['Supplier'].dropna().unique())
        selected_suppliers = st.sidebar.multiselect(
            "🏭 Select Suppliers",
            options=all_suppliers,
            default=[],
            help="Leave empty to include all suppliers"
        )
    else:
        selected_suppliers = []
    
    # Recipient filter
    if 'Recipient' in df.columns:
        all_recipients = sorted(df['Recipient'].dropna().unique())
        selected_recipients = st.sidebar.multiselect(
            "📦 Select Recipients",
            options=all_recipients,
            default=[],
            help="Leave empty to include all recipients"
        )
    else:
        selected_recipients = []
    
    # Weapon category filter
    if 'Weapon Category' in df.columns:
        all_categories = sorted(df['Weapon Category'].dropna().unique())
        selected_categories = st.sidebar.multiselect(
            "🎯 Select Weapon Categories",
            options=all_categories,
            default=[],
            help="Filter by weapon type"
        )
    else:
        selected_categories = []
    
    # Apply filters with improved logic
    filtered_df = df.copy()
    
    if year_range and 'Delivery year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Delivery year'] >= year_range[0]) & 
            (filtered_df['Delivery year'] <= year_range[1])
        ]
    
    if selected_suppliers and 'Supplier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Supplier'].isin(selected_suppliers)]
    
    if selected_recipients and 'Recipient' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Recipient'].isin(selected_recipients)]
    
    if selected_categories and 'Weapon Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Weapon Category'].isin(selected_categories)]
    
    # Display filter status
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()
    
    # ==================== MAIN DASHBOARD TABS ====================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔄 Bilateral Flows", 
        "🔫 Weapons Analysis", 
        "📅 Temporal Trends",
        "🔎 Data Explorer"
    ])
    
    # ==================== TAB 1: OVERVIEW ====================
    
    with tab1:
        st.header("Global Arms Transfer Overview")
        
        # Info message about TIV
        st.info("💡 **Remember:** TIV (Trend Indicator Value) measures military capability transferred, not financial value!")
        
        # Key metrics with black background
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tiv = filtered_df['TIV delivery values'].sum() if 'TIV delivery values' in filtered_df.columns else 0
            st.metric("Total TIV", f"{total_tiv:,.0f}", 
                     help="Sum of all Trend Indicator Values in filtered data")
        
        with col2:
            num_transfers = len(filtered_df)
            st.metric("Number of Transfers", f"{num_transfers:,}")
        
        with col3:
            num_suppliers = filtered_df['Supplier'].nunique() if 'Supplier' in filtered_df.columns else 0
            st.metric("Active Suppliers", f"{num_suppliers:,}")
        
        with col4:
            num_recipients = filtered_df['Recipient'].nunique() if 'Recipient' in filtered_df.columns else 0
            st.metric("Active Recipients", f"{num_recipients:,}")
        
        st.markdown("---")
        
        # NEW: Top Recipients Indicator
        st.subheader("🏆 Top TIV Recipients & Suppliers")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_top_suppliers_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Supplier chart not available - check if data contains Supplier and TIV columns")
        
        with col2:
            fig = create_top_recipients_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⚠️ **Note:** Recipients represent countries importing arms, indicating defense procurement needs")
            else:
                st.info("📊 Recipient data not available - TIV recipient analysis requires 'Recipient' and 'TIV delivery values' columns")
        
        # NEW: Combined comparison
        st.markdown("---")
        fig_comparison = create_top_countries_comparison(filtered_df, n=10)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Time series and categories
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_time_series_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Time series not available - check if data contains year and TIV columns")
        
        with col2:
            fig = create_weapon_category_pie(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Category chart not available")
    
    # ==================== TAB 2: BILATERAL FLOWS ====================
    
    with tab2:
        st.header("Bilateral Arms Transfer Flows")
        st.markdown("*Explore relationships between suppliers and recipients*")
        
        # Heatmap
        st.subheader("Transfer Heatmap")
        top_n = st.slider("Number of top countries to show", 5, 25, 15, 
                         help="Adjust to show more or fewer countries in the matrix")
        
        fig = create_bilateral_heatmap(filtered_df, top_n)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Darker colors indicate higher TIV values (more transfers)")
        else:
            st.info("Heatmap requires Supplier, Recipient, and TIV columns")
    
    # ==================== TAB 3: WEAPONS ANALYSIS ====================
    
    with tab3:
        st.header("Weapons System Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top weapon systems
            if 'Designation' in filtered_df.columns and 'TIV delivery values' in filtered_df.columns:
                st.subheader("Top Weapon Systems by TIV")
                top_weapons = filtered_df.groupby('Designation')['TIV delivery values'].sum().nlargest(15).reset_index()
                
                fig = px.bar(
                    top_weapons,
                    x='TIV delivery values',
                    y='Designation',
                    orientation='h',
                    title="Most Transferred Weapon Systems",
                    color='TIV delivery values',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Weapon system data not available")
        
        with col2:
            # Armament categories
            if 'Armament category' in filtered_df.columns:
                st.subheader("Armament Categories Distribution")
                category_counts = filtered_df['Armament category'].value_counts().head(10)
                
                fig = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation='h',
                    title="Transfer Count by Armament Category",
                    labels={'x': 'Number of Transfers', 'y': 'Category'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Armament category data not available")
    
    # ==================== TAB 4: TEMPORAL TRENDS ====================
    
    with tab4:
        st.header("Temporal Analysis")
        
        if 'Delivery year' in filtered_df.columns:
            # Year selection with error handling
            valid_years = sorted(filtered_df['Delivery year'].dropna().unique(), reverse=True)
            
            if len(valid_years) > 0:
                selected_year = st.selectbox(
                    "Select a year for detailed analysis",
                    valid_years
                )
                
                year_data = filtered_df[filtered_df['Delivery year'] == selected_year]
                
                # Metrics for selected year
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    year_tiv = year_data['TIV delivery values'].sum() if 'TIV delivery values' in year_data.columns else 0
                    st.metric(f"TIV in {int(selected_year)}", f"{year_tiv:,.0f}")
                
                with col2:
                    year_transfers = len(year_data)
                    st.metric(f"Transfers", f"{year_transfers:,}")
                
                with col3:
                    year_suppliers = year_data['Supplier'].nunique() if 'Supplier' in year_data.columns else 0
                    st.metric(f"Suppliers", f"{year_suppliers:,}")
                
                with col4:
                    year_recipients = year_data['Recipient'].nunique() if 'Recipient' in year_data.columns else 0
                    st.metric(f"Recipients", f"{year_recipients:,}")
                
                st.markdown("---")
                
                # Top transfers for selected year
                if all(col in year_data.columns for col in ['Supplier', 'Recipient', 'Designation', 'TIV delivery values']):
                    st.subheader(f"Major Arms Transfers in {int(selected_year)}")
                    
                    # Get columns that exist in the data
                    display_cols = [col for col in ['Supplier', 'Recipient', 'Designation', 'Description', 
                                                    'TIV delivery values', 'Status'] if col in year_data.columns]
                    
                    if len(year_data) > 0:
                        year_major = year_data.nlargest(min(15, len(year_data)), 'TIV delivery values')[display_cols].copy()
                        
                        # Format for display
                        if 'TIV delivery values' in year_major.columns:
                            year_major['TIV delivery values'] = year_major['TIV delivery values'].apply(lambda x: f"{x:,.2f}")
                        
                        st.dataframe(
                            year_major,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No data available for the selected year")
            else:
                st.warning("No valid year data available")
        else:
            st.info("Year data not available for temporal analysis")
    
    # ==================== TAB 5: DATA EXPLORER - FIXED ====================
    
    with tab5:
        st.header("Data Explorer")
        st.markdown("*Search and explore the detailed data*")
        
        # Search functionality - FIXED
        search_term = st.text_input("🔍 Search in data (searches all text columns)", "")
        
        # Start with filtered data
        display_df = filtered_df.copy()
        
        if search_term and len(display_df) > 0:
            # Create mask with same index as display_df
            mask = pd.Series([False] * len(display_df), index=display_df.index)
            
            # Search across all string columns
            for col in display_df.select_dtypes(include=['object']).columns:
                col_mask = display_df[col].astype(str).str.contains(search_term, case=False, na=False)
                mask = mask | col_mask
            
            # Apply mask only if it has matches
            if mask.any():
                display_df = display_df[mask]
        
        # Column selection
        available_cols = list(display_df.columns)
        default_cols = [col for col in ['Supplier', 'Recipient', 'Designation', 'Description', 
                                        'Delivery year', 'TIV delivery values', 'Status', 
                                        'Numbers delivered', 'Armament category'] 
                       if col in available_cols]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            options=available_cols,
            default=default_cols[:7]  # Limit default to 7 columns for readability
        )
        
        if selected_cols and len(display_df) > 0:
            # Display info
            st.info(f"Showing {len(display_df)} records ({len(display_df)/len(df)*100:.1f}% of total data)")
            
            # Display table with pagination
            rows_per_page = st.slider("Rows per page", 10, 100, 25)
            
            total_pages = max(1, len(display_df) // rows_per_page + (1 if len(display_df) % rows_per_page > 0 else 0))
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(display_df))
            
            # Show the dataframe
            st.dataframe(
                display_df[selected_cols].iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            st.markdown("---")
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = display_df[selected_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download filtered data as CSV",
                    data=csv,
                    file_name='sipri_filtered_data.csv',
                    mime='text/csv'
                )
            
            with col2:
                st.info(f"Export will include {len(display_df)} records with {len(selected_cols)} columns")
            
            # Summary statistics for numeric columns
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            numeric_cols = display_df[selected_cols].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write(display_df[numeric_cols].describe())
            else:
                st.info("No numeric columns selected for statistics")
        elif len(display_df) == 0:
            st.warning("No data matches your search criteria")
        else:
            st.info("Please select columns to display")

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()
