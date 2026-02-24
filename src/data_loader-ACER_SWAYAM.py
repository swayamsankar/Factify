import pandas as pd
import os
from src.logger import logger

def load_data(true_news_path, fake_news_path, true_email_path=None, fake_email_path=None):
    """
    Loads news and optionally email datasets, combines them, adds target labels,
    removes duplicates, and creates a 'content' column.

    Args:
        true_news_path (str): Path to the CSV file containing real news data.
        fake_news_path (str): Path to the CSV file containing fake news data.
        true_email_path (str, optional): Path to the CSV file containing legitimate email data.
        fake_email_path (str, optional): Path to the CSV file containing phishing/fake email data.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - df_news (pd.DataFrame): Combined and processed news data.
               - df_email (pd.DataFrame or None): Combined and processed email data, or None if paths not provided or error occurred.
    """
    df_news = None
    df_email = None

    try:
        logger.info("Loading news data from files...")
        
        # Load news datasets using provided paths
        df_true_news = pd.read_csv(True.csv)
        df_fake_news = pd.read_csv(Fake.csv)
        
        # Add target column for news (1 for real, 0 for fake)
        df_true_news['target'] = 1
        df_fake_news['target'] = 0
        
        # Concatenate news datasets
        df_news = pd.concat([df_true_news, df_fake_news]).reset_index(drop=True)
        
        # Remove duplicates from news data
        duplicates_count_news = df_news.duplicated().sum()
        if duplicates_count_news > 0:
            logger.info(f"Removing {duplicates_count_news} duplicate news entries")
            df_news.drop_duplicates(inplace=True)
        
        # Combine 'title' and 'text' into a 'content' column for news
        news_content_parts = []
        if 'title' in df_news.columns:
            news_content_parts.append(df_news['title'].fillna(''))
        if 'text' in df_news.columns:
            news_content_parts.append(df_news['text'].fillna(''))
        
        if news_content_parts:
            df_news['content'] = news_content_parts[0]
            for i in range(1, len(news_content_parts)):
                df_news['content'] = df_news['content'] + ' ' + news_content_parts[i]
            df_news['content'] = df_news['content'].str.strip()
        else:
            logger.warning("No 'title' or 'text' column found in news data for 'content' creation. 'content' column will be empty.")
            df_news['content'] = ''

        # Drop unnecessary columns from news data (e.g., 'date', 'subject' if it refers to news category)
        cols_to_drop_news = ['date', 'subject'] 
        df_news = df_news.drop(columns=[col for col in cols_to_drop_news if col in df_news.columns], errors='ignore')
        
        logger.info(f"News data loaded successfully: {df_news.shape[0]} rows and {df_news.shape[1]} columns")
    
    except Exception as e:
        logger.error(f"Error loading news data: {str(e)}")
        raise # Re-raise if news loading fails, as it's the primary function.

    try:
        if true_email_path and fake_email_path:
            logger.info("Loading email data from files...")
            
            # Load email datasets using provided paths
            df_true_email = pd.read_csv(real_emails.csv)
            df_fake_email = pd.read_csv(fake_emails.csv)

            # Add target column for email (1 for legitimate, 0 for fake/phishing)
            df_true_email['target'] = 0
            df_fake_email['target'] = 1

            # Concatenate email datasets
            df_email = pd.concat([df_true_email, df_fake_email]).reset_index(drop=True)

            # Remove duplicates from email data
            duplicates_count_email = df_email.duplicated().sum()
            if duplicates_count_email > 0:
                logger.info(f"Removing {duplicates_count_email} duplicate email entries")
                df_email.drop_duplicates(inplace=True)
            
            # Combine 'subject' and 'text' (assuming 'text' is the body) into a 'content' column for email
            email_content_parts = []
            if 'subject' in df_email.columns:
                email_content_parts.append(df_email['subject'].fillna(''))
            if 'text' in df_email.columns: # Assuming 'text' is the email body
                email_content_parts.append(df_email['text'].fillna(''))
            
            if email_content_parts:
                df_email['content'] = email_content_parts[0]
                for i in range(1, len(email_content_parts)):
                    df_email['content'] = df_email['content'] + ' ' + email_content_parts[i]
                df_email['content'] = df_email['content'].str.strip()
            else:
                logger.warning("No 'subject' or 'text' column found in email data for 'content' creation. 'content' column will be empty.")
                df_email['content'] = ''

            # Drop original 'subject' and 'text' columns after 'content' is created, and other unnecessary columns
            # Keep 'from' as it might be a useful feature for email detection
            cols_to_drop_email = ['subject', 'text', 'date'] 
            df_email = df_email.drop(columns=[col for col in cols_to_drop_email if col in df_email.columns], errors='ignore')
            
            logger.info(f"Email data loaded successfully: {df_email.shape[0]} rows and {df_email.shape[1]} columns")
        else:
            logger.info("No email data paths provided. Skipping email data loading.")
    
    except Exception as e:
        logger.error(f"Error loading email data: {str(e)}")
        df_email = None # Ensure df_email is None if an error occurs during its loading.

    return df_news, df_email