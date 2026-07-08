from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

css_content = open('style.css', 'r', encoding='utf-8').read()

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(
    Section(open('documentatie.md', 'r', encoding='utf-8').read()),
    user_css=css_content
)
pdf.save('documentatie.pdf')
print("Successfully generated beautifully formatted PDF.")
