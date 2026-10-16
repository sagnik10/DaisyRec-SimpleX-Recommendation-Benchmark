from pathlib import Path
for path in Path('scripts').glob('*.sh'):
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b''))
