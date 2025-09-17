"""
Sample Data Generator for FinScope
Creates realistic financial data for testing and demonstration purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional


class SampleDataGenerator:
    """Class for generating sample financial data with customization options"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            seed: Random seed for reproducible data
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Define categories for expenses
        self.expense_categories = [
            'Marketing', 'Operations', 'Human Resources', 'Technology',
            'Administration', 'Sales', 'Research & Development', 'Legal',
            'Utilities', 'Office Supplies', 'Travel', 'Training'
        ]
        
        # Define revenue sources
        self.revenue_sources = [
            'Product Sales', 'Service Revenue', 'Subscription Fees',
            'Consulting', 'Licensing', 'Partnership Revenue'
        ]
        
        # Business scenarios
        self.business_scenarios = {
            'startup': {
                'revenue_range': (10000, 50000),
                'expense_ratio': 0.85,
                'growth_rate': 0.15,
                'volatility': 0.3,
                'seasonality': 0.2
            },
            'small_business': {
                'revenue_range': (50000, 200000),
                'expense_ratio': 0.75,
                'growth_rate': 0.08,
                'volatility': 0.2,
                'seasonality': 0.3
            },
            'enterprise': {
                'revenue_range': (500000, 2000000),
                'expense_ratio': 0.65,
                'growth_rate': 0.05,
                'volatility': 0.1,
                'seasonality': 0.1
            },
            'non_profit': {
                'revenue_range': (100000, 500000),
                'expense_ratio': 0.95,
                'growth_rate': 0.03,
                'volatility': 0.25,
                'seasonality': 0.4
            }
        }
    
    def generate_custom_data(self, 
                           data_volume: str = 'medium',
                           time_range: str = '2_years',
                           business_scenario: str = 'small_business',
                           currency: str = 'USD',
                           missing_data: float = 0.0,
                           seasonality: str = 'moderate') -> pd.DataFrame:
        """
        Generate customized financial data based on user preferences
        
        Args:
            data_volume: 'small' (100 records), 'medium' (500), 'large' (1000), 'xlarge' (5000)
            time_range: '1_year', '2_years', '3_years', '5_years'
            business_scenario: 'startup', 'small_business', 'enterprise', 'non_profit'
            currency: 'USD', 'EUR', 'GBP', 'INR', 'JPY'
            missing_data: Percentage of missing data (0.0 to 0.2)
            seasonality: 'none', 'moderate', 'high'
            
        Returns:
            DataFrame with customized financial data
        """
        # Map data volume to number of records
        volume_map = {
            'small': 100,
            'medium': 500,
            'large': 1000,
            'xlarge': 5000
        }
        
        # Map time range to years
        time_map = {
            '1_year': 1,
            '2_years': 2,
            '3_years': 3,
            '5_years': 5
        }
        
        # Map seasonality to strength
        seasonality_map = {
            'none': 0.0,
            'moderate': 0.3,
            'high': 0.6
        }
        
        num_records = volume_map[data_volume]
        years = time_map[time_range]
        scenario = self.business_scenarios[business_scenario]
        seasonality_strength = seasonality_map[seasonality]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Generate dates
        date_range = pd.date_range(start=start_date, end=end_date, periods=num_records)
        
        # Generate data based on scenario
        base_revenue = np.random.uniform(*scenario['revenue_range'])
        base_expense = base_revenue * scenario['expense_ratio']
        
        data = []
        
        for i, date in enumerate(date_range):
            # Apply growth over time
            growth_factor = 1 + (scenario['growth_rate'] * i / num_records)
            
            # Apply seasonality
            month = date.month
            seasonal_factor = 1 + seasonality_strength * np.sin(2 * np.pi * month / 12)
            
            # Generate revenue with trend and seasonality
            revenue = base_revenue * growth_factor * seasonal_factor * (1 + np.random.normal(0, scenario['volatility']))
            revenue = max(revenue, 1000)  # Ensure positive revenue
            
            # Generate expenses
            expense = base_expense * growth_factor * (1 + np.random.normal(0, scenario['volatility'] * 0.5))
            expense = max(expense, revenue * 0.3)  # Ensure reasonable expense ratio
            
            # Calculate profit
            profit = revenue - expense
            
            # Generate investment (for ROI calculation)
            investment = np.random.uniform(revenue * 0.1, revenue * 0.3)
            
            # Calculate ROI
            roi = (profit / investment) * 100 if investment > 0 else 0
            
            # Calculate expense ratio
            expense_ratio = (expense / revenue) * 100 if revenue > 0 else 0
            
            # Select random category and revenue source
            category = np.random.choice(self.expense_categories)
            revenue_source = np.random.choice(self.revenue_sources)
            
            # Add some missing data if specified
            if np.random.random() < missing_data:
                # Randomly set some fields to NaN
                if np.random.random() < 0.5:
                    revenue = np.nan
                if np.random.random() < 0.5:
                    expense = np.nan
            
            data.append({
                'date': date,
                'revenue': revenue,
                'expense': expense,
                'profit': profit,
                'investment': investment,
                'roi': roi,
                'expense_ratio': expense_ratio,
                'category': category,
                'revenue_source': revenue_source,
                'currency': currency
            })
        
        df = pd.DataFrame(data)
        
        # Add some additional calculated columns
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        return df

    def generate_monthly_data(self, start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
        """
        Generate monthly financial data for multiple years
        
        Args:
            start_year: Starting year for data generation
            end_year: Ending year for data generation
            
        Returns:
            DataFrame with monthly financial data
        """
        data = []
        current_date = datetime(start_year, 1, 1)
        
        # Base values that will grow over time
        base_revenue = 100000
        base_expense = 80000
        base_investment = 20000
        
        while current_date.year <= end_year:
            # Add some seasonality and growth
            year_factor = (current_date.year - start_year + 1) ** 0.8
            month_factor = 1 + 0.2 * np.sin(2 * np.pi * current_date.month / 12)
            
            # Generate revenue with growth trend and seasonality
            revenue = base_revenue * year_factor * month_factor * np.random.normal(1, 0.1)
            
            # Generate expenses with some correlation to revenue
            expense_ratio = 0.7 + 0.1 * np.random.normal(0, 0.1)
            expense = revenue * expense_ratio * np.random.normal(1, 0.05)
            
            # Generate investment (less frequent, larger amounts)
            if np.random.random() < 0.3:  # 30% chance of investment in a month
                investment = base_investment * year_factor * np.random.uniform(0.5, 2.0)
            else:
                investment = 0
            
            # Calculate profit
            profit = revenue - expense
            
            # Add some random category for expenses
            expense_category = random.choice(self.expense_categories)
            revenue_source = random.choice(self.revenue_sources)
            
            data.append({
                'date': current_date,
                'year': current_date.year,
                'month': current_date.month,
                'revenue': max(0, revenue),
                'expense': max(0, expense),
                'investment': max(0, investment),
                'profit': profit,
                'category': expense_category,
                'revenue_source': revenue_source,
                'description': f"{expense_category} expenses for {current_date.strftime('%B %Y')}"
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return pd.DataFrame(data)
    
    def generate_quarterly_data(self, start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
        """
        Generate quarterly financial data
        
        Args:
            start_year: Starting year for data generation
            end_year: Ending year for data generation
            
        Returns:
            DataFrame with quarterly financial data
        """
        data = []
        
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                # Base values with growth
                year_factor = (year - start_year + 1) ** 0.8
                quarter_factor = 1 + 0.15 * np.sin(2 * np.pi * quarter / 4)
                
                # Generate quarterly revenue
                revenue = 300000 * year_factor * quarter_factor * np.random.normal(1, 0.08)
                
                # Generate quarterly expenses
                expense_ratio = 0.75 + 0.1 * np.random.normal(0, 0.05)
                expense = revenue * expense_ratio * np.random.normal(1, 0.05)
                
                # Generate quarterly investment
                if np.random.random() < 0.4:  # 40% chance of investment in a quarter
                    investment = 60000 * year_factor * np.random.uniform(0.5, 1.5)
                else:
                    investment = 0
                
                profit = revenue - expense
                
                data.append({
                    'date': datetime(year, quarter * 3, 1),
                    'year': year,
                    'quarter': quarter,
                    'revenue': max(0, revenue),
                    'expense': max(0, expense),
                    'investment': max(0, investment),
                    'profit': profit,
                    'category': random.choice(self.expense_categories),
                    'description': f"Q{quarter} {year} financial data"
                })
        
        return pd.DataFrame(data)
    
    def generate_annual_data(self, start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
        """
        Generate annual financial data
        
        Args:
            start_year: Starting year for data generation
            end_year: Ending year for data generation
            
        Returns:
            DataFrame with annual financial data
        """
        data = []
        
        for year in range(start_year, end_year + 1):
            # Base values with growth
            year_factor = (year - start_year + 1) ** 0.9
            
            # Generate annual revenue
            revenue = 1200000 * year_factor * np.random.normal(1, 0.1)
            
            # Generate annual expenses
            expense_ratio = 0.72 + 0.05 * np.random.normal(0, 0.03)
            expense = revenue * expense_ratio * np.random.normal(1, 0.05)
            
            # Generate annual investment
            investment = 240000 * year_factor * np.random.uniform(0.8, 1.2)
            
            profit = revenue - expense
            
            data.append({
                'date': datetime(year, 12, 31),
                'year': year,
                'revenue': max(0, revenue),
                'expense': max(0, expense),
                'investment': max(0, investment),
                'profit': profit,
                'category': 'Annual Operations',
                'description': f"Annual financial summary for {year}"
            })
        
        return pd.DataFrame(data)
    
    def generate_detailed_expense_data(self, start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
        """
        Generate detailed expense data with categories
        
        Args:
            start_year: Starting year for data generation
            end_year: Ending year for data generation
            
        Returns:
            DataFrame with detailed expense data
        """
        data = []
        current_date = datetime(start_year, 1, 1)
        
        while current_date.year <= end_year:
            # Generate monthly data
            monthly_revenue = 100000 * (1 + 0.1 * (current_date.year - start_year)) * np.random.normal(1, 0.1)
            
            # Distribute expenses across categories
            total_expense = monthly_revenue * np.random.uniform(0.65, 0.85)
            
            # Allocate expenses to categories
            category_weights = np.random.dirichlet(np.ones(len(self.expense_categories)))
            category_expenses = total_expense * category_weights
            
            for i, category in enumerate(self.expense_categories):
                if category_expenses[i] > 1000:  # Only include significant expenses
                    data.append({
                        'date': current_date,
                        'year': current_date.year,
                        'month': current_date.month,
                        'revenue': monthly_revenue,
                        'expense': category_expenses[i],
                        'investment': 0,
                        'profit': monthly_revenue - total_expense,
                        'category': category,
                        'description': f"{category} expenses for {current_date.strftime('%B %Y')}"
                    })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return pd.DataFrame(data)
    
    def create_sample_files(self, output_dir: str = "sample_data"):
        """
        Create sample data files for testing
        
        Args:
            output_dir: Directory to save sample files
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate different types of sample data
        monthly_data = self.generate_monthly_data()
        quarterly_data = self.generate_quarterly_data()
        annual_data = self.generate_annual_data()
        detailed_data = self.generate_detailed_expense_data()
        
        # Save to Excel files
        with pd.ExcelWriter(f"{output_dir}/sample_financial_data.xlsx", engine='openpyxl') as writer:
            monthly_data.to_excel(writer, sheet_name='Monthly', index=False)
            quarterly_data.to_excel(writer, sheet_name='Quarterly', index=False)
            annual_data.to_excel(writer, sheet_name='Annual', index=False)
            detailed_data.to_excel(writer, sheet_name='Detailed_Expenses', index=False)
        
        # Save to CSV files
        monthly_data.to_csv(f"{output_dir}/monthly_data.csv", index=False)
        quarterly_data.to_csv(f"{output_dir}/quarterly_data.csv", index=False)
        annual_data.to_csv(f"{output_dir}/annual_data.csv", index=False)
        detailed_data.to_csv(f"{output_dir}/detailed_expenses.csv", index=False)
        
        print(f"Sample data files created in '{output_dir}' directory:")
        print("- sample_financial_data.xlsx (Excel file with multiple sheets)")
        print("- monthly_data.csv")
        print("- quarterly_data.csv")
        print("- annual_data.csv")
        print("- detailed_expenses.csv")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of generated data
        
        Args:
            df: Generated DataFrame
            
        Returns:
            Dictionary with data summary
        """
        return {
            'total_records': len(df),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'total_revenue': df['revenue'].sum(),
            'total_expenses': df['expense'].sum(),
            'total_investment': df['investment'].sum(),
            'net_profit': df['profit'].sum(),
            'average_roi': (df['profit'].sum() / df['investment'].sum() * 100) if df['investment'].sum() > 0 else 0,
            'categories': df['category'].nunique() if 'category' in df.columns else 0
        }


if __name__ == "__main__":
    # Generate sample data when script is run directly
    generator = SampleDataGenerator()
    generator.create_sample_files()
    
    # Show summary of generated data
    monthly_data = generator.generate_monthly_data()
    summary = generator.get_data_summary(monthly_data)
    
    print("\nSample Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
