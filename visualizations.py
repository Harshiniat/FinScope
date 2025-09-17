"""
Visualization Module for FinScope
Creates interactive charts and graphs for financial analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


class FinancialVisualizer:
    """Class for creating financial visualizations"""
    
    def __init__(self, df: pd.DataFrame, currency_symbol: str = '$'):
        """
        Initialize visualizer with DataFrame
        
        Args:
            df: DataFrame containing financial data
            currency_symbol: Currency symbol to use in visualizations
        """
        self.df = df.copy()
        self.currency_symbol = currency_symbol
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting parameters"""
        # Set figure size and style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Set color palette
        self.colors = {
            'revenue': '#2E8B57',      # Sea Green
            'expense': '#DC143C',      # Crimson
            'profit': '#4169E1',       # Royal Blue
            'roi': '#FF8C00',          # Dark Orange
            'investment': '#9370DB',   # Medium Purple
            'expense_ratio': '#FF6347' # Tomato
        }
    
    def create_revenue_expense_trend(self, date_col: str = 'date', 
                                   revenue_col: str = 'revenue', 
                                   expense_col: str = 'expense') -> go.Figure:
        """
        Create line chart showing revenue vs expense trends over time
        
        Args:
            date_col: Name of date column
            revenue_col: Name of revenue column
            expense_col: Name of expense column
            
        Returns:
            Plotly figure object
        """
        # Check if required columns exist
        if revenue_col not in self.df.columns or expense_col not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Required columns '{revenue_col}' or '{expense_col}' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Error: Missing Required Columns",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        fig = go.Figure()
        
        # Add revenue line
        fig.add_trace(go.Scatter(
            x=self.df[date_col] if date_col in self.df.columns else self.df.index,
            y=self.df[revenue_col],
            mode='lines+markers',
            name='Revenue',
            line=dict(color=self.colors['revenue'], width=3),
            marker=dict(size=6)
        ))
        
        # Add expense line
        fig.add_trace(go.Scatter(
            x=self.df[date_col] if date_col in self.df.columns else self.df.index,
            y=self.df[expense_col],
            mode='lines+markers',
            name='Expenses',
            line=dict(color=self.colors['expense'], width=3),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Revenue vs Expenses Trend',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Time Period',
            yaxis_title=f'Amount ({self.currency_symbol})',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickformat=f'{self.currency_symbol},.0f')
        
        return fig
    
    def create_profit_roi_comparison(self, year_col: str = 'year', 
                                   profit_col: str = 'profit', 
                                   roi_col: str = 'roi') -> go.Figure:
        """
        Create bar chart comparing annual profit and ROI
        
        Args:
            year_col: Name of year column
            profit_col: Name of profit column
            roi_col: Name of ROI column
            
        Returns:
            Plotly figure object
        """
        # Check if required columns exist
        if profit_col not in self.df.columns:
            if 'revenue' in self.df.columns and 'expense' in self.df.columns:
                self.df[profit_col] = self.df['revenue'] - self.df['expense']
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Required column '{profit_col}' not found and cannot be calculated",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="red")
                )
                fig.update_layout(
                    title="Error: Missing Required Columns",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                return fig
        
        if roi_col not in self.df.columns:
            if 'investment' in self.df.columns:
                self.df[roi_col] = np.where(
                    self.df['investment'] != 0,
                    (self.df[profit_col] / self.df['investment']) * 100,
                    0
                )
            else:
                self.df[roi_col] = 0
        
        # Group by year
        if year_col in self.df.columns:
            yearly_data = self.df.groupby(year_col).agg({
                profit_col: 'sum',
                roi_col: 'mean'
            }).reset_index()
        else:
            # If no year column, use index
            yearly_data = self.df[[profit_col, roi_col]].copy()
            yearly_data['year'] = range(2020, 2020 + len(yearly_data))
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # Add profit bars
        fig.add_trace(
            go.Bar(
                x=yearly_data[year_col],
                y=yearly_data[profit_col],
                name='Profit',
                marker_color=self.colors['profit'],
                opacity=0.8
            ),
            secondary_y=False
        )
        
        # Add ROI line
        fig.add_trace(
            go.Scatter(
                x=yearly_data[year_col],
                y=yearly_data[roi_col],
                mode='lines+markers',
                name='ROI (%)',
                line=dict(color=self.colors['roi'], width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Annual Profit and ROI Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Year',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text=f"Profit ({self.currency_symbol})", secondary_y=False)
        fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
        
        # Format axes
        fig.update_yaxes(tickformat=f'{self.currency_symbol},.0f', secondary_y=False)
        fig.update_yaxes(tickformat='.1f', secondary_y=True)
        
        return fig
    
    def create_expense_breakdown(self, category_col: str = 'category', 
                               expense_col: str = 'expense') -> go.Figure:
        """
        Create pie chart showing expense breakdown by category
        
        Args:
            category_col: Name of category column
            expense_col: Name of expense column
            
        Returns:
            Plotly figure object
        """
        # Check if expense column exists
        if expense_col not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Required column '{expense_col}' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Error: Missing Required Column",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        if category_col not in self.df.columns:
            # If no category column, create a simple pie chart with total expenses
            fig = go.Figure(data=[go.Pie(
                labels=['Total Expenses'],
                values=[self.df[expense_col].sum()],
                marker_colors=[self.colors['expense']]
            )])
            title = 'Total Expenses'
        else:
            # Group by category
            category_data = self.df.groupby(category_col)[expense_col].sum().reset_index()
            category_data = category_data.sort_values(expense_col, ascending=False)
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=category_data[category_col],
                values=category_data[expense_col],
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )])
            title = 'Expense Breakdown by Category'
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_financial_metrics_dashboard(self, year_col: str = 'year') -> go.Figure:
        """
        Create a comprehensive dashboard with multiple financial metrics
        
        Args:
            year_col: Name of year column
            
        Returns:
            Plotly figure object
        """
        # Ensure required columns exist
        if 'revenue' not in self.df.columns or 'expense' not in self.df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Required columns 'revenue' or 'expense' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Error: Missing Required Columns",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Create missing columns if they don't exist
        if 'profit' not in self.df.columns:
            self.df['profit'] = self.df['revenue'] - self.df['expense']
        
        if 'roi' not in self.df.columns:
            if 'investment' in self.df.columns:
                self.df['roi'] = np.where(
                    self.df['investment'] != 0,
                    (self.df['profit'] / self.df['investment']) * 100,
                    0
                )
            else:
                self.df['roi'] = 0
        
        if 'expense_ratio' not in self.df.columns:
            self.df['expense_ratio'] = np.where(
                self.df['revenue'] != 0,
                (self.df['expense'] / self.df['revenue']) * 100,
                0
            )
        
        # Group by year
        if year_col in self.df.columns:
            yearly_data = self.df.groupby(year_col).agg({
                'revenue': 'sum',
                'expense': 'sum',
                'profit': 'sum',
                'roi': 'mean',
                'expense_ratio': 'mean'
            }).reset_index()
        else:
            yearly_data = self.df[['revenue', 'expense', 'profit', 'roi', 'expense_ratio']].copy()
            yearly_data['year'] = range(2020, 2020 + len(yearly_data))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue & Expenses', 'Profit Trend', 'ROI Trend', 'Expense Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Revenue & Expenses (top left)
        fig.add_trace(
            go.Bar(x=yearly_data[year_col], y=yearly_data['revenue'], 
                   name='Revenue', marker_color=self.colors['revenue']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=yearly_data[year_col], y=yearly_data['expense'], 
                   name='Expenses', marker_color=self.colors['expense']),
            row=1, col=1
        )
        
        # Profit Trend (top right)
        fig.add_trace(
            go.Scatter(x=yearly_data[year_col], y=yearly_data['profit'], 
                      mode='lines+markers', name='Profit', 
                      line=dict(color=self.colors['profit'], width=3)),
            row=1, col=2
        )
        
        # ROI Trend (bottom left)
        fig.add_trace(
            go.Scatter(x=yearly_data[year_col], y=yearly_data['roi'], 
                      mode='lines+markers', name='ROI (%)', 
                      line=dict(color=self.colors['roi'], width=3)),
            row=2, col=1
        )
        
        # Expense Ratio (bottom right)
        fig.add_trace(
            go.Scatter(x=yearly_data[year_col], y=yearly_data['expense_ratio'], 
                      mode='lines+markers', name='Expense Ratio (%)', 
                      line=dict(color=self.colors['expense_ratio'], width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Financial Metrics Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=2)
        fig.update_yaxes(title_text=f"Amount ({self.currency_symbol})", row=1, col=1)
        fig.update_yaxes(title_text=f"Profit ({self.currency_symbol})", row=1, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
        fig.update_yaxes(title_text="Ratio (%)", row=2, col=2)
        
        return fig
    
    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create correlation heatmap for financial metrics
        
        Returns:
            Plotly figure object
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Financial Metrics Correlation Matrix',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=600,
            width=600
        )
        
        return fig
    
    def create_yoy_growth_chart(self, year_col: str = 'year', 
                              revenue_col: str = 'revenue') -> go.Figure:
        """
        Create Year-over-Year growth chart
        
        Args:
            year_col: Name of year column
            revenue_col: Name of revenue column
            
        Returns:
            Plotly figure object
        """
        if year_col not in self.df.columns:
            return go.Figure()
        
        # Calculate YoY growth
        yearly_revenue = self.df.groupby(year_col)[revenue_col].sum()
        yoy_growth = yearly_revenue.pct_change() * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add growth bars
        colors = ['green' if x > 0 else 'red' for x in yoy_growth.values]
        
        fig.add_trace(go.Bar(
            x=yoy_growth.index,
            y=yoy_growth.values,
            marker_color=colors,
            text=[f'{x:.1f}%' for x in yoy_growth.values],
            textposition='auto'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Year-over-Year Revenue Growth',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_savings_analysis_chart(self, expense_col: str = 'expense', 
                                    revenue_col: str = 'revenue') -> go.Figure:
        """
        Create chart showing potential savings opportunities
        
        Args:
            expense_col: Name of expense column
            revenue_col: Name of revenue column
            
        Returns:
            Plotly figure object
        """
        # Check if required columns exist
        if expense_col not in self.df.columns or revenue_col not in self.df.columns:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Required columns '{expense_col}' or '{revenue_col}' not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Error: Missing Required Columns",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Calculate expense ratio
        expense_ratio = (self.df[expense_col] / self.df[revenue_col]) * 100
        avg_ratio = expense_ratio.mean()
        
        # Identify high expense periods
        high_expense_mask = expense_ratio > avg_ratio * 1.1
        
        # Create figure
        fig = go.Figure()
        
        # Add all periods
        fig.add_trace(go.Scatter(
            x=list(range(len(self.df))),
            y=expense_ratio,
            mode='markers',
            name='All Periods',
            marker=dict(color='lightblue', size=8)
        ))
        
        # Add high expense periods
        high_expense_indices = high_expense_mask[high_expense_mask].index.tolist()
        fig.add_trace(go.Scatter(
            x=high_expense_indices,
            y=expense_ratio[high_expense_mask].values,
            mode='markers',
            name='High Expense Periods',
            marker=dict(color='red', size=10)
        ))
        
        # Add average line
        fig.add_hline(y=avg_ratio, line_dash="dash", 
                     line_color="orange", 
                     annotation_text=f"Average: {avg_ratio:.1f}%")
        
        # Add savings threshold line
        savings_threshold = avg_ratio * 1.1
        fig.add_hline(y=savings_threshold, line_dash="dot", 
                     line_color="red", 
                     annotation_text=f"Savings Threshold: {savings_threshold:.1f}%")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Expense Ratio Analysis - Savings Opportunities',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Period',
            yaxis_title='Expense Ratio (%)',
            template='plotly_white',
            height=500
        )
        
        return fig
