"""
Generate PDF Report from Markdown Files
Converts the final project report and other documentation to PDF format
"""

import subprocess
import sys
from pathlib import Path

def check_pandoc():
    """Check if pandoc is installed"""
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def install_markdown_pdf():
    """Install markdown-pdf package"""
    print("Installing markdown-pdf...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown-pdf'],
                      check=True)
        return True
    except:
        return False

def convert_to_pdf_simple(md_file, output_file):
    """
    Simple conversion using Python libraries
    """
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add CSS styling
        css_style = """
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 30px;
            }
            h3 {
                color: #7f8c8d;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
            }
            pre {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            blockquote {
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin-left: 0;
                color: #555;
            }
            .emoji {
                font-size: 1.2em;
            }
        </style>
        """
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=full_html).write_pdf(output_file)
        return True
        
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Installing required libraries...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'markdown', 'weasyprint'], check=True)
            print("Libraries installed! Please run this script again.")
            return False
        except:
            return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def generate_reports():
    """Generate PDF reports from markdown files"""
    project_root = Path(__file__).parent
    
    # Files to convert
    files_to_convert = [
        ('FINAL_PROJECT_REPORT.md', 'FINAL_PROJECT_REPORT.pdf'),
        ('PROJECT_SUMMARY.md', 'PROJECT_SUMMARY.pdf'),
        ('REQUIREMENTS_CHECKLIST.md', 'REQUIREMENTS_CHECKLIST.pdf'),
        ('QUICKSTART.md', 'QUICKSTART.pdf'),
    ]
    
    print("="*70)
    print("PDF REPORT GENERATOR")
    print("="*70)
    print()
    
    success_count = 0
    
    for md_file, pdf_file in files_to_convert:
        md_path = project_root / md_file
        pdf_path = project_root / pdf_file
        
        if not md_path.exists():
            print(f"⚠️  Skipping {md_file} (file not found)")
            continue
        
        print(f"Converting {md_file} to PDF...")
        
        if convert_to_pdf_simple(md_path, pdf_path):
            print(f"✅ Created: {pdf_file}")
            success_count += 1
        else:
            print(f"❌ Failed: {pdf_file}")
        
        print()
    
    print("="*70)
    print(f"Successfully converted {success_count}/{len(files_to_convert)} files")
    print("="*70)
    
    if success_count > 0:
        print("\n✅ PDF reports generated successfully!")
        print("\nGenerated files:")
        for _, pdf_file in files_to_convert[:success_count]:
            print(f"  - {pdf_file}")
    else:
        print("\n❌ No PDF files were generated.")
        print("\nAlternative methods:")
        print("1. Use an online converter: https://www.markdowntopdf.com/")
        print("2. Use VS Code extension: 'Markdown PDF'")
        print("3. Use pandoc (if installed): pandoc input.md -o output.pdf")

if __name__ == '__main__':
    generate_reports()
