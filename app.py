"""
FinScope - Financial Analysis Tool
Main Streamlit application for financial data analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from financial_calculations import FinancialAnalyzer
from data_processor import DataProcessor
from visualizations import FinancialVisualizer
from sample_data_generator import SampleDataGenerator

# Page configuration
st.set_page_config(
    page_title="FinScope Financial Analysis Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def format_currency(amount, currency_symbol=None):
    """Format amount with the selected currency symbol"""
    if currency_symbol is None:
        currency_symbol = st.session_state.get('currency_symbol', '$')
    
    if isinstance(amount, (int, float)):
        return f"{currency_symbol}{amount:,.0f}"
    else:
        return f"{currency_symbol}{amount}"

def main():
    """Main application function"""
    
    # Initialize session state first
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None
    if 'currency_symbol' not in st.session_state:
        st.session_state.currency_symbol = '$'
    
    # Header
    st.markdown('<h1 class="main-header">FinScope Financial Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Currency selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Currency Settings")
    currency_options = {
        '$': 'USD - US Dollar',
        '€': 'EUR - Euro', 
        '£': 'GBP - British Pound',
        '¥': 'JPY - Japanese Yen',
        '₹': 'INR - Indian Rupee',
        '₽': 'RUB - Russian Ruble',
        '₩': 'KRW - South Korean Won',
        '₪': 'ILS - Israeli Shekel',
        '₦': 'NGN - Nigerian Naira',
        '₡': 'CRC - Costa Rican Colon',
        '₱': 'PHP - Philippine Peso',
        '₫': 'VND - Vietnamese Dong',
        '₴': 'UAH - Ukrainian Hryvnia',
        '₸': 'KZT - Kazakhstani Tenge',
        '₼': 'AZN - Azerbaijani Manat',
        '₾': 'GEL - Georgian Lari',
        '₿': 'BTC - Bitcoin',
        'Ξ': 'ETH - Ethereum'
    }
    
    selected_currency = st.sidebar.selectbox(
        "Select Currency:",
        options=list(currency_options.keys()),
        index=list(currency_options.keys()).index(st.session_state.currency_symbol),
        format_func=lambda x: f"{x} - {currency_options[x]}"
    )
    
    if selected_currency != st.session_state.currency_symbol:
        st.session_state.currency_symbol = selected_currency
        st.rerun()
    
    # Handle navigation from buttons
    if 'navigate_to' in st.session_state:
        st.session_state.current_page = st.session_state.navigate_to
        del st.session_state.navigate_to
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Data Analysis", "Visualizations", "Reports", "Data Filtering", "Comparison Mode", "Sample Data"],
        index=["Home", "Data Analysis", "Visualizations", "Reports", "Data Filtering", "Comparison Mode", "Sample Data"].index(st.session_state.current_page)
    )
    
    # Update current page if changed via sidebar
    if page != st.session_state.current_page:
        st.session_state.current_page = page
    
    # Route to different pages
    if st.session_state.current_page == "Home":
        show_home_page()
    elif st.session_state.current_page == "Data Analysis":
        show_analysis_page()
    elif st.session_state.current_page == "Visualizations":
        show_visualizations_page()
    elif st.session_state.current_page == "Reports":
        show_reports_page()
    elif st.session_state.current_page == "Sample Data":
        show_sample_data_page()
    elif st.session_state.current_page == "Data Filtering":
        show_data_filtering_page()
    elif st.session_state.current_page == "Comparison Mode":
        show_comparison_mode_page()

def show_home_page():
    """Display the home page with modern design and project focus"""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 3rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <h1 style="color: white; font-size: 3.5rem; margin-bottom: 1rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">FinScope</h1>
        <p style="color: white; font-size: 1.3rem; margin-bottom: 0; opacity: 0.95; font-weight: 300;">Advanced Financial Analysis Platform</p>
        <p style="color: white; font-size: 1rem; margin-top: 0.5rem; opacity: 0.8;">Built by Harshini</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; margin-bottom: 3rem; border-left: 5px solid #667eea;">
        <h2 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;">Project Overview</h2>
        <p style="font-size: 1.1rem; color: #34495e; line-height: 1.6; margin: 0;">
            FinScope is a comprehensive financial analysis platform that transforms raw financial data into actionable insights. 
            Built with Python and Streamlit, it provides automated calculations, interactive visualizations, and detailed reporting 
            capabilities for businesses and financial analysts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Grid
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; border-top: 3px solid #667eea;">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.2rem;">Financial Analytics</h4>
            <ul style="color: #7f8c8d; margin: 0; padding-left: 1.2rem;">
                <li>ROI & Profit Margin Calculations</li>
                <li>Expense Ratio Analysis</li>
                <li>Year-over-Year Growth Tracking</li>
                <li>Financial Trend Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; border-top: 3px solid #e74c3c;">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.2rem;">Interactive Visualizations</h4>
            <ul style="color: #7f8c8d; margin: 0; padding-left: 1.2rem;">
                <li>Dynamic Line & Bar Charts</li>
                <li>Pie Charts & Heatmaps</li>
                <li>Financial Dashboards</li>
                <li>Real-time Data Filtering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; border-top: 3px solid #27ae60;">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.2rem;">Data Processing</h4>
            <ul style="color: #7f8c8d; margin: 0; padding-left: 1.2rem;">
                <li>Excel & CSV File Support</li>
                <li>Automatic Data Validation</li>
                <li>Smart Column Mapping</li>
                <li>Data Cleaning & Preprocessing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; border-top: 3px solid #f39c12;">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.2rem;">Advanced Features</h4>
            <ul style="color: #7f8c8d; margin: 0; padding-left: 1.2rem;">
                <li>Multi-Currency Support</li>
                <li>Data Comparison Mode</li>
                <li>Comprehensive Reporting</li>
                <li>Sample Data Generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Start Section
    st.markdown("### Get Started")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Start Analysis", type="primary", width='stretch', use_container_width=True):
            st.session_state.navigate_to = "Data Analysis"
            st.rerun()
    
    with col2:
        if st.button("View Sample Data", type="secondary", width='stretch', use_container_width=True):
            st.session_state.navigate_to = "Sample Data"
            st.rerun()
    
    with col3:
        if st.button("View Reports", type="secondary", width='stretch', use_container_width=True):
            st.session_state.navigate_to = "Reports"
            st.rerun()
    
    # How It Works
    st.markdown("---")
    st.markdown("### How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div style="background: #667eea; color: white; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; font-size: 1.8rem; font-weight: bold; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);">1</div>
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.3rem;">Upload Data</h4>
            <p style="color: #7f8c8d; font-size: 0.95rem; line-height: 1.5; margin: 0;">Upload your Excel or CSV financial data files</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div style="background: #667eea; color: white; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; font-size: 1.8rem; font-weight: bold; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);">2</div>
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.3rem;">Analyze & Process</h4>
            <p style="color: #7f8c8d; font-size: 0.95rem; line-height: 1.5; margin: 0;">Automated calculations and data processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div style="background: #667eea; color: white; width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; font-size: 1.8rem; font-weight: bold; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);">3</div>
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem; font-size: 1.3rem;">Visualize Results</h4>
            <p style="color: #7f8c8d; font-size: 0.95rem; line-height: 1.5; margin: 0;">Interactive charts and detailed reports</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; margin-top: 3rem;">
        <h4 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.4rem;">Ready to Transform Your Financial Data?</h4>
        <p style="color: #7f8c8d; margin-bottom: 1.5rem; font-size: 1.1rem;">Start analyzing your financial data in minutes with FinScope</p>
    </div>
    """, unsafe_allow_html=True)

def show_analysis_page():
    """Display the data analysis page"""
    
    st.markdown("## Financial Data Analysis")
    st.markdown("Upload your financial data and configure the analysis parameters.")
    
    # File upload section
    st.markdown("### Upload Financial Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload Excel or CSV files containing your financial data"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Process the uploaded file
        try:
            # Initialize data processor
            processor = DataProcessor()
            
            # Debug information
            st.info(f"Processing file: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size} bytes")
            
            # Validate file
            validation_result = processor.validate_file(uploaded_file)
            
            if validation_result['is_valid']:
                st.success("File uploaded successfully!")
                
                # Load data
                df = processor.load_data(uploaded_file)
                
                # Show data preview
                st.markdown("### Data Preview")
                
                # Show all data option
                show_all_data = st.checkbox("Show All Records", value=False, help="Check this to display all records (may be slow for large datasets)")
                
                if show_all_data:
                    st.dataframe(df, width='stretch')
                else:
                    st.dataframe(df.head(100), width='stretch')
                
                # Show data info
                st.markdown("### Data Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Column mapping section
                st.markdown("### Column Configuration")
                st.markdown("Map your data columns to standard financial metrics:")
                
                # Detect columns automatically
                detected_columns = processor.detect_financial_columns(df)
                
                # Create column mapping interface
                column_mapping = {}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Required Columns:**")
                    revenue_col = st.selectbox(
                        "Revenue Column:",
                        options=[''] + list(df.columns),
                        index=0 if not detected_columns['revenue'] else list(df.columns).index(detected_columns['revenue'][0]) + 1,
                        help="Select the column containing revenue data"
                    )
                    if revenue_col:
                        column_mapping[revenue_col] = 'revenue'
                    
                    expense_col = st.selectbox(
                        "Expense Column:",
                        options=[''] + list(df.columns),
                        index=0 if not detected_columns['expense'] else list(df.columns).index(detected_columns['expense'][0]) + 1,
                        help="Select the column containing expense data"
                    )
                    if expense_col:
                        column_mapping[expense_col] = 'expense'
                
                with col2:
                    st.markdown("**Optional Columns:**")
                    investment_col = st.selectbox(
                        "Investment Column:",
                        options=[''] + list(df.columns),
                        index=0 if not detected_columns['investment'] else list(df.columns).index(detected_columns['investment'][0]) + 1,
                        help="Select the column containing investment data (optional)"
                    )
                    if investment_col:
                        column_mapping[investment_col] = 'investment'
                    
                    date_col = st.selectbox(
                        "Date Column:",
                        options=[''] + list(df.columns),
                        index=0 if not detected_columns['date'] else list(df.columns).index(detected_columns['date'][0]) + 1,
                        help="Select the column containing date data (optional)"
                    )
                    if date_col:
                        column_mapping[date_col] = 'date'
                
                # Run analysis button
                if st.button("Run Financial Analysis", type="primary", width='stretch'):
                    if revenue_col and expense_col:
                        with st.spinner("Processing data and running analysis..."):
                            try:
                                # Map columns
                                df_mapped = processor.map_columns(df, column_mapping)
                                
                                # Prepare data for analysis
                                df_processed = processor.prepare_for_analysis(df_mapped)
                                
                                # Run financial analysis
                                analyzer = FinancialAnalyzer(df_processed)
                                analysis_results = analyzer.run_full_analysis()
                                
                                # Store results in session state
                                st.session_state.analysis_results = analysis_results
                                st.session_state.processed_data = df_processed
                                
                                st.success("✅ Analysis completed successfully! You can now explore the results in the 'Data Analysis' and 'Visualizations' tabs.")
                                
                                # Show key metrics
                                st.markdown("### Key Financial Metrics")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    total_revenue = df_processed['revenue'].sum()
                                    st.metric("Total Revenue", format_currency(total_revenue))
                                
                                with col2:
                                    total_expense = df_processed['expense'].sum()
                                    st.metric("Total Expenses", format_currency(total_expense))
                                
                                with col3:
                                    total_profit = df_processed['profit'].sum()
                                    st.metric("Total Profit", format_currency(total_profit))
                                
                                with col4:
                                    avg_roi = analysis_results['roi'].mean() if not analysis_results['roi'].empty else 0
                                    st.metric("Average ROI", f"{avg_roi:.1f}%")
                                
                                # Show savings analysis
                                if 'savings_analysis' in analysis_results:
                                    savings = analysis_results['savings_analysis']
                                    if 'potential_savings' in savings:
                                        st.markdown("### Savings Opportunities")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Potential Savings", format_currency(savings['potential_savings']))
                                        with col2:
                                            st.metric("Savings Percentage", f"{savings['savings_percentage']:.1f}%")
                                        with col3:
                                            st.metric("High Expense Periods", savings['high_expense_periods'])
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                    else:
                        st.error("Please select both Revenue and Expense columns to proceed.")
            
            else:
                st.error(f"{validation_result['message']}")
                if validation_result['suggestions']:
                    st.warning(f"{validation_result['suggestions']}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_visualizations_page():
    """Display the visualizations page"""
    
    st.markdown("## Financial Visualizations")
    
    if st.session_state.analysis_results is None:
        st.warning("Please run the financial analysis first by going to the 'Data Analysis' page.")
        return
    
    # Get analysis results
    analysis_results = st.session_state.analysis_results
    df = analysis_results['data']
    
    # Visualization options
    st.markdown("### Choose Visualization Type")
    
    viz_type = st.selectbox(
        "Select visualization:",
        [
            "Revenue vs Expenses Trend",
            "Annual Profit and ROI Comparison", 
            "Expense Breakdown by Category",
            "Financial Metrics Dashboard",
            "Correlation Heatmap",
            "Year-over-Year Growth",
            "Savings Opportunities Analysis"
        ]
    )
    
    # Create visualizer
    visualizer = FinancialVisualizer(df, st.session_state.currency_symbol)
    
    # Generate selected visualization
    try:
        if viz_type == "Revenue vs Expenses Trend":
            fig = visualizer.create_revenue_expense_trend()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Annual Profit and ROI Comparison":
            fig = visualizer.create_profit_roi_comparison()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Expense Breakdown by Category":
            fig = visualizer.create_expense_breakdown()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Financial Metrics Dashboard":
            fig = visualizer.create_financial_metrics_dashboard()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Correlation Heatmap":
            fig = visualizer.create_correlation_heatmap()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Year-over-Year Growth":
            fig = visualizer.create_yoy_growth_chart()
            st.plotly_chart(fig, width='stretch')
            
        elif viz_type == "Savings Opportunities Analysis":
            fig = visualizer.create_savings_analysis_chart()
            st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.info("Try using the sample data generator to test the visualizations with sample data.")
    
    # Additional insights
    st.markdown("### Insights")
    
    if 'savings_analysis' in analysis_results:
        savings = analysis_results['savings_analysis']
        if 'potential_savings' in savings:
            st.info(f"""
            **Savings Opportunity**: Based on the analysis, there's potential to save 
            **{format_currency(savings['potential_savings'])}** by optimizing expenses in high-expense periods. 
            This represents approximately **{savings['savings_percentage']:.1f}%** of the identified high-expense periods.
            """)

def show_reports_page():
    """Display the reports page"""
    
    st.markdown("## Financial Reports")
    
    if st.session_state.analysis_results is None:
        st.warning("Please run the financial analysis first by going to the 'Data Analysis' page.")
        return
    
    # Get analysis results
    analysis_results = st.session_state.analysis_results
    df = analysis_results['data']
    
    # Annual summary report
    st.markdown("### Annual Financial Summary")
    
    if not analysis_results['annual_summary'].empty:
        st.dataframe(analysis_results['annual_summary'], width='stretch')
        
        # Export options
        st.markdown("### Export Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export to CSV
            csv = analysis_results['annual_summary'].to_csv()
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"financial_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export to Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                analysis_results['annual_summary'].to_excel(writer, sheet_name='Annual Summary')
                df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name=f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Export detailed analysis
            detailed_report = create_detailed_report(analysis_results)
            st.download_button(
                label="Download Detailed Report",
                data=detailed_report,
                file_name=f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Key metrics summary
    st.markdown("### Key Metrics Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Financial Performance:**
        - Total Revenue: {format_currency(df['revenue'].sum())}
        - Total Expenses: {format_currency(df['expense'].sum())}
        - Net Profit: {format_currency(df['profit'].sum())}
        - Average ROI: {analysis_results['roi'].mean() if not analysis_results['roi'].empty else 0:.1f}%
        """)
    
    with col2:
        if 'savings_analysis' in analysis_results:
            savings = analysis_results['savings_analysis']
            st.markdown(f"""
            **Savings Opportunities:**
            - Potential Savings: {format_currency(savings.get('potential_savings', 0))}
            - Savings Percentage: {savings.get('savings_percentage', 0):.1f}%
            - High Expense Periods: {savings.get('high_expense_periods', 0)}
            - Average Expense Ratio: {savings.get('average_expense_ratio', 0):.1f}%
            """)

def show_sample_data_page():
    """Display the sample data generation page with customization options"""
    
    st.markdown("## Sample Data Generation")
    st.markdown("Generate realistic financial data with customizable parameters to test FinScope's capabilities.")
    
    # Debug: Check if sample data exists
    if 'sample_data' in st.session_state and st.session_state.sample_data is not None:
        st.info(f"Sample data already exists in session state. Shape: {st.session_state.sample_data.shape}")
    else:
        st.info("No sample data in session state yet.")
    
    # Customization options
    st.markdown("### Customization Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Configuration")
        
        data_volume = st.selectbox(
            "Data Volume:",
            ["small", "medium", "large", "xlarge"],
            index=1,
            help="Small: 100 records, Medium: 500, Large: 1000, XLarge: 5000"
        )
        
        time_range = st.selectbox(
            "Time Range:",
            ["1_year", "2_years", "3_years", "5_years"],
            index=1,
            help="Select the time period for data generation"
        )
        
        business_scenario = st.selectbox(
            "Business Scenario:",
            ["startup", "small_business", "enterprise", "non_profit"],
            index=1,
            help="Choose a business type that affects revenue patterns and growth rates"
        )
        
        currency = st.selectbox(
            "Currency:",
            ["USD", "EUR", "GBP", "INR", "JPY"],
            index=0,
            help="Select the currency for financial data"
        )
    
    with col2:
        st.markdown("#### Data Quality Settings")
        
        missing_data = st.slider(
            "Missing Data (%):",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Percentage of missing values to simulate real-world data quality issues"
        )
        
        seasonality = st.selectbox(
            "Seasonality:",
            ["none", "moderate", "high"],
            index=1,
            help="Level of seasonal variation in the data"
        )
        
        # Show scenario description
        scenario_descriptions = {
            "startup": "High growth, high volatility, moderate seasonality",
            "small_business": "Steady growth, moderate volatility, some seasonality",
            "enterprise": "Stable growth, low volatility, minimal seasonality",
            "non_profit": "Slow growth, high volatility, high seasonality"
        }
        
        st.markdown(f"**Scenario Description:** {scenario_descriptions[business_scenario]}")
    
    # Generate sample data button
    if st.button("Generate Custom Sample Data", type="primary", width='stretch'):
        try:
            generator = SampleDataGenerator()
            sample_data = generator.generate_custom_data(
                data_volume=data_volume,
                time_range=time_range,
                business_scenario=business_scenario,
                currency=currency,
                missing_data=missing_data/100,  # Convert percentage to decimal
                seasonality=seasonality
            )
            
            # Store in session state
            st.session_state.sample_data = sample_data
            
            st.success(f"Custom sample data generated successfully! ({len(sample_data)} records)")
            
            # Debug information
            st.info(f"Sample data stored in session state. Shape: {sample_data.shape}")
            
            # Show comprehensive data display
            st.markdown("---")
            st.markdown("## Generated Sample Data")
            
            # Data overview
            st.markdown("### Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(sample_data))
            with col2:
                st.metric("Date Range", f"{sample_data['date'].min().strftime('%Y-%m-%d')} to {sample_data['date'].max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Total Revenue", format_currency(sample_data['revenue'].sum()))
            with col4:
                st.metric("Total Expenses", format_currency(sample_data['expense'].sum()))
            
            # Financial metrics
            st.markdown("### Financial Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Net Profit", format_currency(sample_data['profit'].sum()))
            with col2:
                st.metric("Average ROI", f"{sample_data['roi'].mean():.1f}%")
            with col3:
                st.metric("Average Expense Ratio", f"{sample_data['expense_ratio'].mean():.1f}%")
            with col4:
                st.metric("Profit Margin", f"{(sample_data['profit'].sum() / sample_data['revenue'].sum() * 100):.1f}%")
            
            # Data quality metrics
            st.markdown("### Data Quality")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Categories", sample_data['category'].nunique())
            with col2:
                st.metric("Revenue Sources", sample_data['revenue_source'].nunique())
            with col3:
                st.metric("Missing Values", sample_data.isnull().sum().sum())
            with col4:
                st.metric("Data Completeness", f"{((len(sample_data) - sample_data.isnull().sum().sum()) / (len(sample_data) * len(sample_data.columns)) * 100):.1f}%")
            
            # Data preview with tabs
            st.markdown("### Data Preview")
            
            # Show all data option
            show_all = st.checkbox("Show All Records", value=False, help="Check this to display all records (may be slow for large datasets)")
            
            if show_all:
                st.dataframe(sample_data, width='stretch')
            else:
                tab1, tab2, tab3 = st.tabs(["First 100 Records", "Last 100 Records", "Random Sample (100)"])
                
                with tab1:
                    st.dataframe(sample_data.head(100), width='stretch')
                
                with tab2:
                    st.dataframe(sample_data.tail(100), width='stretch')
                
                with tab3:
                    st.dataframe(sample_data.sample(min(100, len(sample_data))), width='stretch')
            
            # Column information
            st.markdown("### Column Information")
            col_info = []
            for col in sample_data.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(sample_data[col].dtype),
                    'Non-Null Count': sample_data[col].count(),
                    'Null Count': sample_data[col].isnull().sum(),
                    'Unique Values': sample_data[col].nunique()
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, width='stretch')
            
            # Category breakdown
            if 'category' in sample_data.columns:
                st.markdown("### Category Breakdown")
                category_summary = sample_data.groupby('category').agg({
                    'revenue': 'sum',
                    'expense': 'sum',
                    'profit': 'sum',
                    'roi': 'mean'
                }).round(2)
                # Keep raw numbers for dataframe display
                # category_summary['revenue'] = category_summary['revenue'].apply(lambda x: format_currency(x))
                # category_summary['expense'] = category_summary['expense'].apply(lambda x: format_currency(x))
                # category_summary['profit'] = category_summary['profit'].apply(lambda x: format_currency(x))
                # category_summary['roi'] = category_summary['roi'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(category_summary, width='stretch')
            
            # Download options
            st.markdown("### Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"sample_data_{data_volume}_{business_scenario}_{len(sample_data)}_records.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                # Create Excel file
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    sample_data.to_excel(writer, sheet_name='Sample Data', index=False)
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Records', 'Date Range', 'Total Revenue', 'Total Expenses', 'Net Profit', 'Average ROI', 'Average Expense Ratio'],
                        'Value': [
                            len(sample_data),
                            f"{sample_data['date'].min().strftime('%Y-%m-%d')} to {sample_data['date'].max().strftime('%Y-%m-%d')}",
                            sample_data['revenue'].sum(),
                            sample_data['expense'].sum(),
                            sample_data['profit'].sum(),
                            f"{sample_data['roi'].mean():.1f}%",
                            f"{sample_data['expense_ratio'].mean():.1f}%"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="Download as Excel",
                    data=output.getvalue(),
                    file_name=f"sample_data_{data_volume}_{business_scenario}_{len(sample_data)}_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary"
                )
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating sample data: {str(e)}")
    
    # Add a separate button outside the generation block for using sample data
    if 'sample_data' in st.session_state and st.session_state.sample_data is not None:
        st.markdown("---")
        st.markdown("### Use Generated Data")
        if st.button("Use This Data for Analysis", type="secondary", key="use_sample_data"):
            try:
                # Check if sample data exists
                if st.session_state.sample_data is None:
                    st.error("No sample data available. Please generate sample data first.")
                    return
                
                # Get sample data from session state
                sample_data = st.session_state.sample_data
                
                # Process the sample data
                processor = DataProcessor()
                df_processed = processor.prepare_for_analysis(sample_data)
                
                # Run analysis
                analyzer = FinancialAnalyzer(df_processed)
                analysis_results = analyzer.run_full_analysis()
                
                # Store in session state
                st.session_state.analysis_results = analysis_results
                st.session_state.processed_data = df_processed
                
                # Show success notification with redirect info
                st.success("Sample data loaded and analyzed successfully! Redirecting to Visualizations page...")
                
                # Redirect to Visualizations page
                st.session_state.current_page = "Visualizations"
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing sample data: {str(e)}")

def create_detailed_report(analysis_results):
    """Create a detailed text report"""
    
    report = []
    report.append("FINANCIAL ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data summary
    df = analysis_results['data']
    report.append("DATA SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Records: {len(df)}")
    report.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    report.append("")
    
    # Financial metrics
    report.append("FINANCIAL METRICS")
    report.append("-" * 20)
    currency_symbol = st.session_state.get('currency_symbol', '$')
    report.append(f"Total Revenue: {currency_symbol}{df['revenue'].sum():,.2f}")
    report.append(f"Total Expenses: {currency_symbol}{df['expense'].sum():,.2f}")
    report.append(f"Net Profit: {currency_symbol}{df['profit'].sum():,.2f}")
    report.append(f"Average ROI: {analysis_results['roi'].mean():.2f}%")
    report.append("")
    
    # Savings analysis
    if 'savings_analysis' in analysis_results:
        savings = analysis_results['savings_analysis']
        report.append("SAVINGS OPPORTUNITIES")
        report.append("-" * 20)
        report.append(f"Potential Savings: {currency_symbol}{savings.get('potential_savings', 0):,.2f}")
        report.append(f"Savings Percentage: {savings.get('savings_percentage', 0):.1f}%")
        report.append(f"High Expense Periods: {savings.get('high_expense_periods', 0)}")
        report.append("")
    
    return "\n".join(report)

def show_data_filtering_page():
    """Display the data filtering page"""
    
    st.markdown("## 🔍 Data Filtering")
    st.markdown("Filter and analyze your financial data with advanced filtering options.")
    
    if st.session_state.processed_data is None:
        st.warning("No data available for filtering. Please upload data or generate sample data first.")
        return
    
    df = st.session_state.processed_data.copy()
    
    # Filter options
    st.markdown("### Filter Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range filter
        if 'date' in df.columns:
            st.markdown("#### Date Range")
            date_col = st.selectbox("Select date column:", df.columns, key="date_filter_col")
            
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                
                date_range = st.date_input(
                    "Select date range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_filter"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
        
        # Category filter
        if 'category' in df.columns:
            st.markdown("#### Category Filter")
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_categories = st.multiselect(
                "Select categories:",
                categories[1:],  # Exclude 'All'
                default=categories[1:],
                key="category_filter"
            )
            
            if selected_categories:
                df = df[df['category'].isin(selected_categories)]
    
    with col2:
        # Revenue range filter
        if 'revenue' in df.columns:
            st.markdown("#### Revenue Range")
            min_revenue = float(df['revenue'].min())
            max_revenue = float(df['revenue'].max())
            
            revenue_range = st.slider(
                "Select revenue range:",
                min_value=min_revenue,
                max_value=max_revenue,
                value=(min_revenue, max_revenue),
                format=f"{st.session_state.currency_symbol}%.0f",
                key="revenue_range_filter"
            )
            
            df = df[(df['revenue'] >= revenue_range[0]) & (df['revenue'] <= revenue_range[1])]
        
        # Search filter
        st.markdown("#### Search")
        search_term = st.text_input("Search in data:", key="search_filter")
        
        if search_term:
            # Search in text columns
            text_columns = df.select_dtypes(include=['object']).columns
            mask = df[text_columns].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            df = df[mask]
    
    # Display filtered results
    st.markdown("### Filtered Results")
    
    if len(df) == 0:
        st.warning("No data matches the selected filters.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Records", len(df))
    
    with col2:
        if 'revenue' in df.columns:
            st.metric("Total Revenue", format_currency(df['revenue'].sum()))
    
    with col3:
        if 'expense' in df.columns:
            st.metric("Total Expenses", format_currency(df['expense'].sum()))
    
    with col4:
        if 'profit' in df.columns:
            st.metric("Net Profit", format_currency(df['profit'].sum()))
    
    # Data preview
    st.markdown("#### Filtered Data Preview")
    
    # Show all data option
    show_all_filtered = st.checkbox("Show All Filtered Records", value=False, help="Check this to display all filtered records")
    
    if show_all_filtered:
        st.dataframe(df, width='stretch')
    else:
        st.dataframe(df.head(100), width='stretch')
    
    # Download filtered data
    if st.button("Download Filtered Data", type="primary"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_comparison_mode_page():
    """Display the comparison mode page"""
    
    st.markdown("## ⚖️ Comparison Mode")
    st.markdown("Compare two datasets or time periods to analyze differences and trends.")
    
    if st.session_state.processed_data is None:
        st.warning("No data available for comparison. Please upload data or generate sample data first.")
        return
    
    df = st.session_state.processed_data.copy()
    
    # Comparison options
    st.markdown("### Comparison Options")
    
    comparison_type = st.radio(
        "Select comparison type:",
        ["Time Period Comparison", "Custom Split"],
        key="comparison_type"
    )
    
    if comparison_type == "Time Period Comparison":
        if 'date' in df.columns:
            st.markdown("#### Time Period Comparison")
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Get unique years
            years = sorted(df['date'].dt.year.unique())
            
            col1, col2 = st.columns(2)
            
            with col1:
                period1_year = st.selectbox("Select first period year:", years, key="period1_year")
                period1_data = df[df['date'].dt.year == period1_year]
            
            with col2:
                period2_year = st.selectbox("Select second period year:", years, key="period2_year")
                period2_data = df[df['date'].dt.year == period2_year]
            
            if period1_year != period2_year:
                # Calculate metrics for both periods
                metrics1 = calculate_period_metrics(period1_data, "Period 1")
                metrics2 = calculate_period_metrics(period2_data, "Period 2")
                
                # Display comparison
                display_comparison_metrics(metrics1, metrics2)
                
                # Create comparison charts
                create_comparison_charts(period1_data, period2_data, f"{period1_year}", f"{period2_year}")
    
    elif comparison_type == "Custom Split":
        st.markdown("#### Custom Data Split")
        
        # Allow user to define custom split criteria
        split_column = st.selectbox("Select column to split by:", df.columns, key="split_column")
        
        if df[split_column].dtype in ['object', 'category']:
            # Categorical split
            unique_values = df[split_column].unique()
            
            col1, col2 = st.columns(2)
            
            with col1:
                split1_values = st.multiselect("Select values for Group 1:", unique_values, key="split1_values")
                group1_data = df[df[split_column].isin(split1_values)]
            
            with col2:
                split2_values = st.multiselect("Select values for Group 2:", unique_values, key="split2_values")
                group2_data = df[df[split_column].isin(split2_values)]
            
            if split1_values and split2_values:
                # Calculate metrics for both groups
                metrics1 = calculate_period_metrics(group1_data, "Group 1")
                metrics2 = calculate_period_metrics(group2_data, "Group 2")
                
                # Display comparison
                display_comparison_metrics(metrics1, metrics2)
                
                # Create comparison charts
                create_comparison_charts(group1_data, group2_data, "Group 1", "Group 2")

def calculate_period_metrics(data, period_name):
    """Calculate key metrics for a period"""
    metrics = {
        'period_name': period_name,
        'total_records': len(data),
        'total_revenue': data['revenue'].sum() if 'revenue' in data.columns else 0,
        'total_expenses': data['expense'].sum() if 'expense' in data.columns else 0,
        'net_profit': data['profit'].sum() if 'profit' in data.columns else 0,
        'avg_roi': data['roi'].mean() if 'roi' in data.columns else 0,
        'avg_expense_ratio': data['expense_ratio'].mean() if 'expense_ratio' in data.columns else 0
    }
    return metrics

def display_comparison_metrics(metrics1, metrics2):
    """Display comparison metrics in a table format"""
    st.markdown("### Comparison Metrics")
    
    comparison_data = {
        'Metric': ['Total Records', 'Total Revenue', 'Total Expenses', 'Net Profit', 'Avg ROI (%)', 'Avg Expense Ratio (%)'],
        metrics1['period_name']: [
            metrics1['total_records'],
            metrics1['total_revenue'],
            metrics1['total_expenses'],
            metrics1['net_profit'],
            metrics1['avg_roi'],
            metrics1['avg_expense_ratio']
        ],
        metrics2['period_name']: [
            metrics2['total_records'],
            metrics2['total_revenue'],
            metrics2['total_expenses'],
            metrics2['net_profit'],
            metrics2['avg_roi'],
            metrics2['avg_expense_ratio']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width='stretch')
    
    # Calculate percentage changes
    st.markdown("### Percentage Changes")
    
    changes = {
        'Metric': ['Revenue Change', 'Expense Change', 'Profit Change', 'ROI Change', 'Expense Ratio Change'],
        'Change (%)': [
            ((metrics2['total_revenue'] - metrics1['total_revenue']) / metrics1['total_revenue'] * 100) if metrics1['total_revenue'] != 0 else 0,
            ((metrics2['total_expenses'] - metrics1['total_expenses']) / metrics1['total_expenses'] * 100) if metrics1['total_expenses'] != 0 else 0,
            ((metrics2['net_profit'] - metrics1['net_profit']) / metrics1['net_profit'] * 100) if metrics1['net_profit'] != 0 else 0,
            ((metrics2['avg_roi'] - metrics1['avg_roi']) / metrics1['avg_roi'] * 100) if metrics1['avg_roi'] != 0 else 0,
            ((metrics2['avg_expense_ratio'] - metrics1['avg_expense_ratio']) / metrics1['avg_expense_ratio'] * 100) if metrics1['avg_expense_ratio'] != 0 else 0
        ]
    }
    
    changes_df = pd.DataFrame(changes)
    st.dataframe(changes_df, width='stretch')

def create_comparison_charts(data1, data2, label1, label2):
    """Create comparison charts"""
    st.markdown("### Comparison Charts")
    
    # Revenue comparison
    if 'revenue' in data1.columns and 'revenue' in data2.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=label1,
            x=[label1],
            y=[data1['revenue'].sum()],
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            name=label2,
            x=[label2],
            y=[data2['revenue'].sum()],
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Revenue Comparison",
            xaxis_title="Period",
            yaxis_title=f"Revenue ({st.session_state.currency_symbol})",
            yaxis=dict(tickformat=f'{st.session_state.currency_symbol},.0f'),
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # Profit comparison
    if 'profit' in data1.columns and 'profit' in data2.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=label1,
            x=[label1],
            y=[data1['profit'].sum()],
            marker_color='#27ae60'
        ))
        
        fig.add_trace(go.Bar(
            name=label2,
            x=[label2],
            y=[data2['profit'].sum()],
            marker_color='#f39c12'
        ))
        
        fig.update_layout(
            title="Profit Comparison",
            xaxis_title="Period",
            yaxis_title=f"Profit ({st.session_state.currency_symbol})",
            yaxis=dict(tickformat=f'{st.session_state.currency_symbol},.0f'),
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()
