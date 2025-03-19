#!/usr/bin/env python3
"""
PDF to Markdown Converter using Gemma 3

This script processes PDF files, converts them to images, and uses Gemma 3 to convert
the images to markdown with accurate text recognition, image descriptions, and context
for hard-to-read text.
"""

import os
import argparse
import concurrent.futures
import requests
import base64
import io
import json
import random
from pathlib import Path
from tqdm import tqdm
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image

input_directory = './2025_jfk_pdfs' 
output_directory = './2025_jfk_mds'
gemma_endpoint = 'http://localhost:5000/v1/chat/completions'  # Hardcoded Gemma 3 API endpoint
dpi = 300  # Hardcoded DPI for PDF to image conversion
workers = 1   # Hardcoded number of worker processes
chunk_size = 10  # Hardcoded number of pages to process at once
password = None  # Hardcoded password for protected PDF files (None if not needed)
prompt_file = None  # Hardcoded path to a text file containing a custom prompt for Gemma 3 (None if not needed)
max_tokens = 10000  # Hardcoded maximum tokens in Gemma 3 response

def convert_pdf_to_images(pdf_path, dpi=300, first_page=None, last_page=None, password=None):
    """
    Convert a PDF file to a list of images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for the resulting images
        first_page: First page to convert (1-based)
        last_page: Last page to convert (1-based)
        password: Password for protected PDF
    
    Returns:
        List of PIL Image objects
    """
    try:
        return convert_from_path(
            pdf_path, 
            dpi=dpi, 
            first_page=first_page, 
            last_page=last_page,
            userpw=password
        )
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def process_image_with_gemma(image, gemma_endpoint, custom_prompt=None, max_tokens=4000, max_retries=3):
    """
    Process an image with the local Gemma 3 model using a REST API.
    
    Args:
        image: PIL Image object
        gemma_endpoint: URL for the Gemma 3 API endpoint
        custom_prompt: Custom prompt to use instead of the default
        max_tokens: Maximum number of tokens in the response
        max_retries: Number of retries for API calls
    
    Returns:
        Markdown content as a string
    """
    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Use custom prompt if provided, otherwise use the default
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = """
        I need you to convert this document image to markdown with extreme accuracy. Follow these instructions carefully:
        
        1. Extract all text with exact formatting preservation:
           - Maintain heading levels, paragraphs, lists, and tables
           - Preserve text alignment, indentation, and spacing where possible
           - Keep the reading order exactly as in the original
        
        2. For any images in the document:
           - Provide a detailed description in markdown format: ![Description of what the image shows](image)
           - Include what the image depicts, its purpose, and any visible text
           - Be specific about charts, graphs, diagrams, photos, etc.
        
        3. For hard-to-read or uncertain text:
           - Mark uncertain text with [?text?] where "text" is your best guess
           - Add contextual information where helpful: [context: appears to be a technical term]
           - For completely illegible text: [illegible: appears to be approximately X words/characters]
        
        4. Special formatting:
           - Use appropriate markdown for all formatting elements:
             * Headings (#, ##, etc.)
             * Lists (-, *, 1., etc.)
             * Tables (| --- |)
             * Block quotes (>)
             * Code blocks (```)
           - Represent mathematical equations and formulas accurately
        
        5. Document structure:
           - Include page headers/footers but mark them as such
           - Note any page numbers as: [Page X]
           - For multi-column layouts, process one column completely before moving to the next
        
        Respond ONLY with the markdown content, nothing else.
        """
    
    # Create API request payload - mimicking OpenAI API format but using requests directly
    payload = {
        "model": "gemma-3-27b-it",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert OCR and document conversion specialist with perfect accuracy. Your task is to convert document images to markdown, extracting all text, describing images, and handling uncertain text with appropriate context."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ],
        "max_tokens": max_tokens
    }
    
    # Try to call the API with retries
    for attempt in range(max_retries):
        try:
            # Send the request to the local Gemma 3 endpoint without timeout
            response = requests.post(gemma_endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Handle both OpenAI-like and custom API formats
                if "choices" in result and len(result["choices"]) > 0:
                    if "message" in result["choices"][0]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        return result["choices"][0]["content"]
                elif "content" in result:
                    return result["content"]
                else:
                    raise Exception(f"Unexpected response format from Gemma 3 API: {result}")
            else:
                print(f"API error (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                if attempt == max_retries - 1:
                    raise Exception(f"Error from Gemma 3 API after {max_retries} attempts")
        except requests.RequestException as e:
            print(f"Request error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to connect to Gemma 3 API after {max_retries} attempts")
    
    # If we got here, all retries failed
    return "Error: Failed to process this page with Gemma 3"


def process_pdf_directory(input_dir, output_dir, gemma_endpoint, dpi=300, workers=4, chunk_size=10, 
                         password=None, custom_prompt=None, max_tokens=4000):
    """
    Process all PDF files in a directory and convert them to markdown.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory where markdown files will be saved
        gemma_endpoint: URL for the Gemma 3 API endpoint
        dpi: DPI for the resulting images
        workers: Number of worker processes for parallel processing
        chunk_size: Number of pages to process at once (to manage memory)
        password: Password for protected PDF files
        custom_prompt: Custom prompt to use for Gemma 3
        max_tokens: Maximum number of tokens in Gemma 3 response
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob('**/*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        # Create relative path structure in output directory
        rel_path = pdf_path.relative_to(input_path)
        output_path = Path(output_dir) / rel_path.with_suffix('.md')
        
        # Skip if output file already exists
        if output_path.exists():
            print(f"Skipping: {rel_path} (output file already exists)")
            continue
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {rel_path}")
        
        try:
            # Get the number of pages
            try:
                pdf = PdfReader(str(pdf_path))
                if pdf.is_encrypted and password:
                    pdf.decrypt(password)
                total_pages = len(pdf.pages)
            except Exception as e:
                print(f"  Error reading PDF: {e}")
                if "password" in str(e).lower():
                    print("  The PDF appears to be password protected. Use the --password option.")
                continue
            
            # Process in chunks to manage memory
            full_markdown = []
            
            # Create a list of all page numbers and shuffle them
            all_pages = list(range(1, total_pages + 1))
            random.shuffle(all_pages)
            
            # Process pages in randomized chunks
            for i in range(0, len(all_pages), chunk_size):
                chunk_pages = all_pages[i:i + chunk_size]
                
                print(f"  Converting pages {chunk_pages} to images")
                
                # Process each page individually since they're now random
                chunk_results = []
                
                for page_num in chunk_pages:
                    # Convert single PDF page to image
                    images = convert_pdf_to_images(
                        str(pdf_path), 
                        dpi=dpi,
                        first_page=page_num,
                        last_page=page_num,
                        password=password
                    )
                    
                    if not images:
                        print(f"  Error: Could not convert page {page_num} to image. Skipping.")
                        chunk_results.append((page_num, f"\n\n## [Page {page_num} - ERROR]\n\nFailed to convert page to image\n\n"))
                        continue
                    
                    # We only requested one page, so there should be only one image
                    image = images[0]
                    chunk_results.append((page_num, image))
                
                # Process pages in parallel
                print(f"  Processing randomized pages with Gemma 3")
                
                # Process pages with a progress bar
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            process_image_with_gemma, 
                            img, 
                            gemma_endpoint, 
                            custom_prompt=custom_prompt,
                            max_tokens=max_tokens
                        ): (page, img) for page, img in chunk_results if isinstance(img, Image.Image)
                    }
                    
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing pages"):
                        page_num, _ = futures[future]
                        try:
                            result = future.result()
                            # Add page number indicator
                            result = f"[Page {page_num}]\n\n{result}"
                            results.append((page_num, result))
                        except Exception as e:
                            error_msg = f"Error processing page {page_num}: {str(e)}"
                            print(f"  {error_msg}")
                            results.append((page_num, f"\n\n## [Page {page_num} - ERROR]\n\n{error_msg}\n\n"))
                
                # Add any error results that were skipped earlier
                for page_num, result in chunk_results:
                    if isinstance(result, str):  # This is an error message
                        results.append((page_num, result))
                
                # Add results to full markdown
                full_markdown.extend([result for _, result in results])
            
            # Sort results by page number for final output
            full_markdown_with_pages = [(int(md.split('[Page ')[1].split(']')[0]), md) for md in full_markdown]
            full_markdown_with_pages.sort(key=lambda x: x[0])
            sorted_markdown = [md for _, md in full_markdown_with_pages]
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n\n---\n\n".join(sorted_markdown))
            
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")


def main():
    process_pdf_directory(
        input_directory, 
        output_directory, 
        gemma_endpoint, 
        dpi=dpi,
        workers=workers,
        chunk_size=chunk_size,
        password=password,
        custom_prompt=prompt_file,
        max_tokens=max_tokens
    )

if __name__ == "__main__":
    main()