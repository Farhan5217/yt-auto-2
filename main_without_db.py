import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types
from pydantic import BaseModel
from supadata import Supadata, SupadataError
from prompt import prompt
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TranscriptResponse(BaseModel):
    text: str

class Config:
    """Configuration from environment variables"""
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    SUPADATA_API_KEY = os.getenv('SUPADATA_API_KEY')
    GOOGLE_CREDENTIALS = os.getenv('GOOGLE_CREDENTIALS')
    SHEET_NAME = os.getenv('SHEET_NAME')
    WORKSHEET_NAME = os.getenv('WORKSHEET_NAME', 'Sheet1')

class VideoProcessor:
    def __init__(self):
        self.validate_environment()
        self.setup_google_sheets()
        self.setup_gemini()
        self.setup_supadata()
        
    def validate_environment(self):
        """Validate all required environment variables are set"""
        required_vars = ['GEMINI_API_KEY', 'SUPADATA_API_KEY', 'GOOGLE_CREDENTIALS', 'SHEET_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {missing_vars}")
            
        logging.info("All environment variables validated")
        
    def setup_google_sheets(self):
        """Setup Google Sheets using credentials from environment"""
        try:
            # Parse credentials from environment variable
            creds_info = json.loads(Config.GOOGLE_CREDENTIALS)
            
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = Credentials.from_service_account_info(creds_info, scopes=scope)
            self.gc = gspread.authorize(creds)
            self.sheet = self.gc.open(Config.SHEET_NAME).worksheet(Config.WORKSHEET_NAME)
            
            logging.info(f"Connected to Google Sheet: {Config.SHEET_NAME}")
            
        except Exception as e:
            logging.error(f"Failed to setup Google Sheets: {e}")
            raise
            
    def setup_gemini(self):
        """Setup Gemini AI client"""
        try:
            self.genai_client = genai.Client(api_key=Config.GEMINI_API_KEY)
            logging.info("Gemini AI client initialized")
        except Exception as e:
            logging.error(f"Failed to setup Gemini: {e}")
            raise
            
    def setup_supadata(self):
        """Setup Supadata client for transcription"""
        try:
            self.supadata = Supadata(api_key=Config.SUPADATA_API_KEY)
            logging.info("Supadata client initialized")
        except Exception as e:
            logging.error(f"Failed to setup Supadata: {e}")
            raise
            
    def is_supported_video_url(self, url: str) -> bool:
        """Check if URL is supported by Supadata"""
        supported_domains = [
            'youtube.com', 'youtu.be',           # YouTube
            'twitter.com', 'x.com',              # Twitter/X
            'vimeo.com',                         # Vimeo
            'tiktok.com',                        # TikTok
            'instagram.com',                     # Instagram
            'facebook.com', 'fb.com',            # Facebook
            'linkedin.com',                      # LinkedIn
            'reddit.com',                        # Reddit
        ]
        
        return any(domain in url.lower() for domain in supported_domains)
        
    def get_transcript(self, url: str) -> Optional[str]:
        """Get transcript using Supadata API"""
        try:
            logging.info(f"Getting transcript for URL: {url}")
            
            transcript_response = self.supadata.transcript(
                url=url,
                lang="en",     # Preferred language
                text=True,     # Return plain text instead of timestamped chunks
                mode="auto"    # Use auto mode for best results
            )
            
            # Handle the supadata.types.Transcript object
            if hasattr(transcript_response, 'content'):
                transcript_text = transcript_response.content
                detected_lang = getattr(transcript_response, 'lang', 'unknown')
                logging.info(f"Transcript retrieved ({len(transcript_text)} characters, language: {detected_lang})")
                return transcript_text
            elif isinstance(transcript_response, dict) and 'content' in transcript_response:
                transcript_text = transcript_response['content']
                detected_lang = transcript_response.get('lang', 'unknown')
                logging.info(f"Transcript retrieved ({len(transcript_text)} characters, language: {detected_lang})")
                return transcript_text
            elif isinstance(transcript_response, str):
                logging.info(f"Transcript retrieved ({len(transcript_response)} characters)")
                return transcript_response
            else:
                # Debug: log all available attributes
                attrs = [attr for attr in dir(transcript_response) if not attr.startswith('_')]
                logging.warning(f"Unknown transcript object. Available attributes: {attrs}")
                
                # Try common attribute names
                for attr_name in ['text', 'transcript', 'data', 'result']:
                    if hasattr(transcript_response, attr_name):
                        content = getattr(transcript_response, attr_name)
                        logging.info(f"Found content in '{attr_name}' attribute")
                        return str(content)
                
                logging.warning(f"Could not extract content from transcript object")
                return None
                
        except SupadataError as e:
            logging.error(f"Supadata error for {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error getting transcript for {url}: {e}")
            return None
        
    def analyze_with_gemini(self, transcript: str, system_prompt: str) -> Optional[str]:
        """Analyze transcript using Gemini AI"""
        try:
            # Truncate transcript if too long (Gemini has token limits)
            max_chars = 100000  # Adjust based on your needs
            if len(transcript) > max_chars:
                transcript = transcript[:max_chars] + "... [TRUNCATED]"
                logging.warning(f"Transcript truncated to {max_chars} characters")
            
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=transcript,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=TranscriptResponse,
                )
            )
            
            result = response.parsed.text
            logging.info("Gemini analysis completed")
            return result
            
        except Exception as e:
            logging.error(f"Gemini analysis failed: {e}")
            return None
            
    def get_new_urls(self) -> List[Dict]:
        """Get new URLs from Google Sheets that haven't been processed"""
        try:
            all_values = self.sheet.get_all_values()
            new_urls = []
            
            for row_idx, row in enumerate(all_values, start=1):
                if len(row) == 0:
                    continue
                    
                url = row[0].strip() if len(row) > 0 else ""
                status = row[2].strip() if len(row) > 2 else ""
                
                # Skip if not a supported video URL
                if not url or not self.is_supported_video_url(url):
                    continue
                    
                # Skip if already processed (has any status)
                if status in ['PROCESSING', 'COMPLETED', 'ERROR'] or status.strip():
                    continue
                    
                new_urls.append({
                    'url': url,
                    'row': row_idx
                })
                
            logging.info(f"Found {len(new_urls)} new URLs to process")
            return new_urls
            
        except Exception as e:
            logging.error(f"Failed to get URLs from sheet: {e}")
            return []
            
    def update_sheet_status(self, row: int, status: str, result: str = ""):
        """Update processing status in Google Sheets"""
        try:
            if result:
                # Truncate result if too long for Google Sheets
                max_cell_chars = 50000
                if len(result) > max_cell_chars:
                    result = result[:max_cell_chars] + "... [TRUNCATED]"
                
                # Update result in column B
                try:
                    self.sheet.update_acell(f'B{row}', result)
                    logging.info(f"Updated result in B{row}")
                except Exception as e:
                    logging.error(f"Failed to update result in B{row}: {e}")
                
            # Update status in column C
            try:
                self.sheet.update_acell(f'C{row}', status)
                logging.info(f"Updated status in C{row}: {status}")
            except Exception as e:
                logging.error(f"Failed to update status in C{row}: {e}")
            
        except Exception as e:
            logging.error(f"Failed to update sheet row {row}: {e}")
            
    def process_url(self, url_data: Dict) -> bool:
        """Process a single video URL"""
        url = url_data['url']
        row = url_data['row']
        
        logging.info(f"Processing URL: {url} (row {row})")
        
        # Update status to PROCESSING
        self.update_sheet_status(row, "PROCESSING")
        
        try:
            # Get transcript using Supadata
            transcript = self.get_transcript(url)
            if not transcript:
                raise Exception("Could not retrieve transcript using Supadata")
                
            logging.info(f"Retrieved transcript ({len(transcript)} characters)")

            
            analysis = self.analyze_with_gemini(transcript,prompt)
            if not analysis:
                raise Exception("Gemini analysis failed")
                
            # Update sheet with results
            self.update_sheet_status(row, "COMPLETED", analysis)
            
            logging.info(f"Successfully processed URL: {url}")
            return True
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Failed to process {url}: {error_msg}")
            
            self.update_sheet_status(row, "ERROR", error_msg)
            return False
            
    def run(self):
        """Main processing loop"""
        logging.info("Starting video automation pipeline")
        
        try:
            new_urls = self.get_new_urls()
            
            if not new_urls:
                logging.info("No new URLs to process")
                return
                
            processed_count = 0
            error_count = 0
            
            for url_data in new_urls:
                try:
                    if self.process_url(url_data):
                        processed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logging.error(f"Unexpected error processing {url_data['url']}: {e}")
                    error_count += 1
                    
            logging.info(f"Processing complete. Success: {processed_count}, Errors: {error_count}")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")

def main():
    """Entry point for cron job"""
    try:
        processor = VideoProcessor()
        processor.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()