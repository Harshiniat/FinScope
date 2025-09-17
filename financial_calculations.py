"""
Financial Calculations Module for FinScope
Handles all financial calculations including ROI, expense ratios, and growth metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FinancialAnalyzer:
    """Main class for financial analysis and calculations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the financial analyzer with a DataFrame
        
        Args:
            df: Pandas DataFrame containing financial data
        """
        self.df = df.copy()
        self.results = {}
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and standardize the financial data
        
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with all NaN values
        self.df = self.df.dropna(how='all')
        
        # Standardize column names (case insensitive)
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Try to identify date column
        date_columns = ['date', 'year', 'month', 'period', 'quarter']
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    break
                except:
                    continue
        
        # Fill missing values with 0 for numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        
        return self.df
    
    def calculate_profit(self, revenue_col: str = 'revenue', expense_col: str = 'expense') -> pd.Series:
        """
        Calculate profit as Revenue - Expense
        
        Args:
            revenue_col: Name of revenue column
            expense_col: Name of expense column
            
        Returns:
            Series containing profit values
        """
        if revenue_col in self.df.columns and expense_col in self.df.columns:
            profit = self.df[revenue_col] - self.df[expense_col]
            self.results['profit'] = profit
            return profit
        else:
            print(f"Warning: Revenue or expense columns not found. Available columns: {list(self.df.columns)}")
            return pd.Series()
    
    def calculate_roi(self, profit_col: str = 'profit', investment_col: str = 'investment') -> pd.Series:
        """
        Calculate ROI as (Profit / Investment) * 100
        
        Args:
            profit_col: Name of profit column
            investment_col: Name of investment column
            
        Returns:
            Series containing ROI values
        """
        if profit_col in self.df.columns and investment_col in self.df.columns:
            # Avoid division by zero
            roi = np.where(
                self.df[investment_col] != 0,
                (self.df[profit_col] / self.df[investment_col]) * 100,
                0
            )
            roi_series = pd.Series(roi, index=self.df.index)
            self.results['roi'] = roi_series
            return roi_series
        else:
            print(f"Warning: Profit or investment columns not found. Available columns: {list(self.df.columns)}")
            return pd.Series()
    
    def calculate_expense_ratio(self, expense_col: str = 'expense', revenue_col: str = 'revenue') -> pd.Series:
        """
        Calculate expense ratio as (Expense / Revenue) * 100
        
        Args:
            expense_col: Name of expense column
            revenue_col: Name of revenue column
            
        Returns:
            Series containing expense ratio values
        """
        if expense_col in self.df.columns and revenue_col in self.df.columns:
            # Avoid division by zero
            expense_ratio = np.where(
                self.df[revenue_col] != 0,
                (self.df[expense_col] / self.df[revenue_col]) * 100,
                0
            )
            ratio_series = pd.Series(expense_ratio, index=self.df.index)
            self.results['expense_ratio'] = ratio_series
            return ratio_series
        else:
            print(f"Warning: Expense or revenue columns not found. Available columns: {list(self.df.columns)}")
            return pd.Series()
    
    def calculate_yoy_growth(self, revenue_col: str = 'revenue', date_col: str = 'date') -> pd.Series:
        """
        Calculate Year-over-Year growth rate
        
        Args:
            revenue_col: Name of revenue column
            date_col: Name of date column
            
        Returns:
            Series containing YoY growth values
        """
        if revenue_col in self.df.columns and date_col in self.df.columns:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            
            # Extract year from date
            self.df['year'] = self.df[date_col].dt.year
            
            # Calculate YoY growth
            yearly_revenue = self.df.groupby('year')[revenue_col].sum()
            yoy_growth = yearly_revenue.pct_change() * 100
            
            # Map back to original dataframe
            self.df['yoy_growth'] = self.df['year'].map(yoy_growth)
            self.results['yoy_growth'] = self.df['yoy_growth']
            
            return self.df['yoy_growth']
        else:
            print(f"Warning: Revenue or date columns not found. Available columns: {list(self.df.columns)}")
            return pd.Series()
    
    def get_annual_summary(self) -> pd.DataFrame:
        """
        Generate annual financial summary
        
        Returns:
            DataFrame with annual summary statistics
        """
        # Try to identify year column
        year_col = None
        for col in ['year', 'date']:
            if col in self.df.columns:
                if col == 'date' and pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    self.df['year'] = self.df[col].dt.year
                    year_col = 'year'
                elif col == 'year':
                    year_col = col
                break
        
        if year_col is None:
            # If no year column, create one based on index
            self.df['year'] = range(2020, 2020 + len(self.df))
            year_col = 'year'
        
        # Group by year and calculate metrics
        summary_cols = []
        for col in ['revenue', 'expense', 'profit', 'investment', 'roi', 'expense_ratio']:
            if col in self.df.columns:
                summary_cols.append(col)
        
        if summary_cols:
            annual_summary = self.df.groupby(year_col)[summary_cols].agg({
                col: ['sum', 'mean'] if col in ['revenue', 'expense', 'profit', 'investment'] else 'mean'
                for col in summary_cols
            }).round(2)
            
            # Flatten column names
            annual_summary.columns = ['_'.join(col).strip() for col in annual_summary.columns]
            
            return annual_summary
        else:
            return pd.DataFrame()
    
    def identify_savings_opportunities(self, expense_col: str = 'expense', revenue_col: str = 'revenue') -> Dict:
        """
        Identify potential savings opportunities
        
        Args:
            expense_col: Name of expense column
            revenue_col: Name of revenue column
            
        Returns:
            Dictionary with savings analysis
        """
        if expense_col not in self.df.columns or revenue_col not in self.df.columns:
            return {"error": "Required columns not found"}
        
        # Calculate average expense ratio
        avg_expense_ratio = (self.df[expense_col].sum() / self.df[revenue_col].sum()) * 100
        
        # Identify high expense periods
        expense_ratio = (self.df[expense_col] / self.df[revenue_col]) * 100
        high_expense_periods = self.df[expense_ratio > avg_expense_ratio * 1.1]
        
        # Calculate potential savings (assuming 8% reduction in high expense periods)
        potential_savings = high_expense_periods[expense_col].sum() * 0.08
        
        return {
            "average_expense_ratio": round(avg_expense_ratio, 2),
            "high_expense_periods": len(high_expense_periods),
            "potential_savings": round(potential_savings, 2),
            "savings_percentage": 8.0
        }
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete financial analysis
        
        Returns:
            Dictionary containing all analysis results
        """
        # Clean data first
        self.clean_data()
        
        # Run all calculations
        profit = self.calculate_profit()
        roi = self.calculate_roi()
        expense_ratio = self.calculate_expense_ratio()
        yoy_growth = self.calculate_yoy_growth()
        
        # Generate summaries
        annual_summary = self.get_annual_summary()
        savings_analysis = self.identify_savings_opportunities()
        
        return {
            "data": self.df,
            "profit": profit,
            "roi": roi,
            "expense_ratio": expense_ratio,
            "yoy_growth": yoy_growth,
            "annual_summary": annual_summary,
            "savings_analysis": savings_analysis,
            "results": self.results
        }
