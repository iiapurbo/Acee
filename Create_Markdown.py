import litellm
import os
from pdf2image import convert_from_path
import PIL.Image
import io
import base64

# Set the Gemini API key
os.environ['GEMINI_API_KEY'] = "AIzaSyDehjBgSUtoDnQ7I4eMlwKBeVfkNMZ3rEs"

# Hardcoded paths
PDF_PATH = "./ace/143.-ITSRO-339-October-2024-Exemption-of-income-of-Grameen-Bank.pdf"
OUTPUT_DIR = "outputs"
POPPLER_PATH = r'C:\Users\RDR\Downloads\Compressed\Release-24.08.0-0\poppler-24.08.0\Library\bin'

def get_response(prompt, image=None):
    """Get response from Gemini 2.0 Flash using LiteLLM with image input."""
    try:
        messages = [{"role": "user", "content": prompt}]
        
        if image:
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_data = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Add image message in Gemini-compatible format
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    }
                ]
            })
        
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def convert_pdf_to_markdown(pdf_path, output_dir, poppler_path):
    """Convert PDF to Markdown with UTF-8 support for Bangla text."""
    try:
        # Verify PDF exists
        if not os.path.exists(pdf_path):
            raise Exception(f"PDF file not found at {pdf_path}")

        # Verify Poppler path
        if not os.path.exists(poppler_path):
            raise Exception(f"Poppler path not found at {poppler_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images with specified DPI and Poppler path
        print(f"Converting {pdf_path} to images...")
        images = convert_from_path(pdf_path, dpi=500, poppler_path=poppler_path)
        
        # Prepare output Markdown file
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        output_path = os.path.join(output_dir, output_filename)
        
        # Prompt for Markdown conversion with Bangla support
        prompt = (
            "Convert the contents of this page into Markdown format, preserving structure and formatting such as headers, lists, tables, and text styling. "
            "Ensure all text, including Bangla (Bengali) script, is accurately transcribed and compatible with UTF-8 encoding."
        )
        
        # Process each page
        with open(output_path, "w", encoding="utf-8") as f:
            # Write UTF-8 BOM for compatibility
            f.write("\ufeff")
            f.write("# Document Conversion\n\n")
            for i, image in enumerate(images):
                print(f"Processing page {i+1}/{len(images)}...")
                markdown_content = get_response(prompt, image)
                f.write(f"\n## Page {i+1}\n\n")
                f.write(markdown_content + "\n")
                print(f"Page {i+1} processed.")
        
        print(f"Markdown file saved as {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error during PDF conversion: {str(e)}")
        return None

def main():
    """Convert the specified PDF to Markdown."""
    print("Starting PDF to Markdown conversion...")
    result = convert_pdf_to_markdown(PDF_PATH, OUTPUT_DIR, POPPLER_PATH)
    if result:
        print(f"Conversion successful! Markdown file: {result}")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()