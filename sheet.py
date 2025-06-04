#!/usr/bin/env python3
"""
Download a Google Sheet as CSV file.

Usage: python sheet.py <google_sheet_url>

The script will download the sheet and save it as a CSV file named after the sheet's title.
"""

import sys
import re
import urllib.parse
import urllib.request
import json
import csv
import os
from dotenv import load_dotenv


def extract_sheet_id_and_gid(url):
    """Extract the spreadsheet ID and gid from a Google Sheets URL."""
    # Extract spreadsheet ID
    sheet_id_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
    if not sheet_id_match:
        raise ValueError("Could not extract spreadsheet ID from URL")
    sheet_id = sheet_id_match.group(1)
    
    # Extract gid (sheet tab ID)
    gid_match = re.search(r'[#&]gid=([0-9]+)', url)
    gid = gid_match.group(1) if gid_match else '0'
    
    return sheet_id, gid


def get_sheet_title(sheet_id, gid):
    """Get the title of the specific sheet tab."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_SHEETS_API_KEY')
    
    if not api_key:
        print("Warning: GOOGLE_SHEETS_API_KEY not found in .env file")
        return 'sheet'
    
    # Use the Google Sheets API to get sheet metadata
    api_url = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}?key={api_key}"
    
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            
        # Find the sheet with matching gid
        for sheet in data.get('sheets', []):
            if str(sheet['properties']['sheetId']) == gid:
                return sheet['properties']['title']
                
        # If no matching gid found, use the first sheet's title
        if data.get('sheets'):
            return data['sheets'][0]['properties']['title']
            
    except Exception:
        # Fallback: try to extract title from the main spreadsheet
        try:
            return data.get('properties', {}).get('title', 'sheet')
        except:
            pass
    
    # Final fallback
    return 'sheet'


def download_sheet_as_csv(sheet_id, gid):
    """Download the Google Sheet as CSV."""
    # Construct the CSV export URL
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    with urllib.request.urlopen(csv_url) as response:
        return response.read().decode('utf-8')


def sanitize_filename(filename):
    """Sanitize filename by removing/replacing invalid characters."""
    # Remove or replace characters that are invalid in filenames
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    # Ensure it's not empty
    if not filename:
        filename = 'sheet'
    return filename


def main():
    if len(sys.argv) != 2:
        print("Usage: python sheet.py <google_sheet_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    try:
        # Extract sheet ID and gid from URL
        sheet_id, gid = extract_sheet_id_and_gid(url)
        print(f"Sheet ID: {sheet_id}, GID: {gid}")
        
        # Get sheet title
        sheet_title = get_sheet_title(sheet_id, gid)
        print(f"Sheet title: {sheet_title}")
        
        # Sanitize the title for use as filename
        safe_filename = sanitize_filename(sheet_title)
        csv_filename = f"{safe_filename}.csv"
        
        # Download the sheet as CSV
        print(f"Downloading sheet...")
        csv_content = download_sheet_as_csv(sheet_id, gid)
        
        # Save to file
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)
        
        print(f"Successfully saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
