"""
Data Processing Module for FinScope
Handles data cleaning, validation, and preparation for financial analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Class for processing and cleaning financial data"""
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls', '.csv']
        self.required_columns = ['revenue', 'expense']
        self.optional_columns = ['investment', 'date', 'year', 'category', 'description']
    
    def validate_file(self, file_object) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate uploaded file format and structure
        
        Args:
            file_object: Streamlit file upload object
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'message': '',
            'columns': [],
            'suggestions': []
        }
        
        try:
            # Check file extension
            file_ext = file_object.name.lower().split('.')[-1]
            if f'.{file_ext}' not in self.supported_formats:
                result['message'] = f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
                return result
            
            # Reset file pointer to beginning
            file_object.seek(0)
            
            # Try to read the file
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_object, nrows=5)  # Read only first 5 rows for validation
            else:
                df = pd.read_csv(file_object, nrows=5)
            
            result['columns'] = list(df.columns)
            result['is_valid'] = True
            result['message'] = "File is valid"
            
            # Check for required columns
            missing_required = []
            column_suggestions = {}
            
            for req_col in self.required_columns:
                found = False
                for col in df.columns:
                    if req_col.lower() in col.lower():
                        column_suggestions[req_col] = col
                        found = True
                        break
                
                if not found:
                    missing_required.append(req_col)
            
            if missing_required:
                result['suggestions'] = f"Missing required columns: {', '.join(missing_required)}"
                result['is_valid'] = False
            else:
                result['suggestions'] = f"Column mapping suggestions: {column_suggestions}"
            
        except Exception as e:
            result['message'] = f"Error reading file: {str(e)}"
        
        return result
    
    def load_data(self, file_object, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file with error handling
        
        Args:
            file_object: Streamlit file upload object
            sheet_name: Sheet name for Excel files (optional)
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_ext = file_object.name.lower().split('.')[-1]
            
            # Reset file pointer to beginning
            file_object.seek(0)
            
            if file_ext in ['xlsx', 'xls']:
                if sheet_name:
                    df = pd.read_excel(file_object, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(file_object)
            else:
                df = pd.read_csv(file_object)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def clean_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize financial data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Handle date columns
        date_columns = ['date', 'year', 'month', 'period', 'quarter', 'timestamp']
        for col in date_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    # If conversion successful, extract year
                    if col == 'date':
                        df_clean['year'] = df_clean[col].dt.year
                    break
                except:
                    continue
        
        # If no date column found, create a year column based on index
        if 'year' not in df_clean.columns:
            df_clean['year'] = range(2020, 2020 + len(df_clean))
        
        # Clean numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Replace negative values with 0 for financial data (unless it's profit/loss)
            if col not in ['profit', 'loss', 'net_income']:
                df_clean[col] = df_clean[col].clip(lower=0)
            # Fill NaN values with 0
            df_clean[col] = df_clean[col].fillna(0)
        
        # Handle text columns
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean
    
    def map_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Map user-specified columns to standard names
        
        Args:
            df: DataFrame with original column names
            column_mapping: Dictionary mapping original names to standard names
            
        Returns:
            DataFrame with mapped column names
        """
        df_mapped = df.copy()
        
        # Rename columns based on mapping
        df_mapped = df_mapped.rename(columns=column_mapping)
        
        return df_mapped
    
    def detect_financial_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect financial columns based on content and names
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with detected column categories
        """
        detected = {
            'revenue': [],
            'expense': [],
            'investment': [],
            'date': [],
            'category': [],
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Revenue indicators
            if any(keyword in col_lower for keyword in ['revenue', 'income', 'sales', 'earnings', 'gross']):
                detected['revenue'].append(col)
            # Expense indicators
            elif any(keyword in col_lower for keyword in ['expense', 'cost', 'spending', 'outlay', 'expenditure']):
                detected['expense'].append(col)
            # Investment indicators
            elif any(keyword in col_lower for keyword in ['investment', 'capital', 'asset', 'purchase']):
                detected['investment'].append(col)
            # Date indicators
            elif any(keyword in col_lower for keyword in ['date', 'year', 'month', 'period', 'time', 'quarter']):
                detected['date'].append(col)
            # Category indicators
            elif any(keyword in col_lower for keyword in ['category', 'type', 'department', 'division', 'class']):
                detected['category'].append(col)
            else:
                detected['other'].append(col)
        
        return detected
    
    def validate_financial_data(self, df: pd.DataFrame) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate financial data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Check for required columns
        if 'revenue' not in df.columns:
            validation['errors'].append("Revenue column not found")
            validation['is_valid'] = False
        
        if 'expense' not in df.columns:
            validation['errors'].append("Expense column not found")
            validation['is_valid'] = False
        
        # Check for data quality issues
        if 'revenue' in df.columns:
            if df['revenue'].isna().all():
                validation['errors'].append("Revenue column contains no valid data")
                validation['is_valid'] = False
            elif df['revenue'].sum() <= 0:
                validation['warnings'].append("Total revenue is zero or negative")
        
        if 'expense' in df.columns:
            if df['expense'].isna().all():
                validation['errors'].append("Expense column contains no valid data")
                validation['is_valid'] = False
            elif df['expense'].sum() <= 0:
                validation['warnings'].append("Total expenses are zero or negative")
        
        # Check for reasonable data ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['revenue', 'expense', 'investment']:
                if df[col].max() > 1e12:  # More than 1 trillion
                    validation['warnings'].append(f"{col} values seem unusually large")
                if df[col].min() < 0 and col != 'profit':
                    validation['warnings'].append(f"{col} contains negative values")
        
        # Check for missing data
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.5]
        if not high_missing.empty:
            validation['warnings'].append(f"High missing data in columns: {list(high_missing.index)}")
        
        return validation
    
    def prepare_for_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for financial analysis
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Prepared DataFrame ready for analysis
        """
        # Clean the data
        df_clean = self.clean_financial_data(df)
        
        # Ensure we have required columns
        if 'revenue' not in df_clean.columns or 'expense' not in df_clean.columns:
            raise ValueError("Data must contain 'revenue' and 'expense' columns")
        
        # Add calculated columns if not present
        if 'profit' not in df_clean.columns:
            df_clean['profit'] = df_clean['revenue'] - df_clean['expense']
        
        # Ensure investment column exists (default to 0 if not present)
        if 'investment' not in df_clean.columns:
            df_clean['investment'] = 0
        
        # Sort by date if available
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date')
        elif 'year' in df_clean.columns:
            df_clean = df_clean.sort_values('year')
        
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'date_range': None,
            'columns': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Date range
        if 'date' in df.columns and not df['date'].isna().all():
            summary['date_range'] = {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
        elif 'year' in df.columns:
            summary['date_range'] = {
                'start': df['year'].min(),
                'end': df['year'].max()
            }
        
        return summary
