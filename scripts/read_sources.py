import zipfile, xml.etree.ElementTree as ET, re
from pathlib import Path
z=zipfile.ZipFile('CA6124 Report v2.docx')
r=ET.fromstring(z.read('word/document.xml'))
s='\n'.join(''.join(p.itertext()) for p in r.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))
Path('artifacts/report-source.txt').write_text(s, encoding='utf8')
print(s)
