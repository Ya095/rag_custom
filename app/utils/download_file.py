import requests
from pathlib import Path


APP_PATH = Path(__file__).parent.parent
OUTPUT_FOLDER = 'documents'
FILENAME = "think_python_guide.pdf"
url = "https://greenteapress.com/thinkpython/thinkpython.pdf"
file_path = APP_PATH / OUTPUT_FOLDER / FILENAME


def download_file(res_url: str, res_file_path: Path) -> None:
    print("Скачивание файла.")
    response = requests.get(res_url, stream=True, timeout=30)
    response.raise_for_status()
    res_file_path.parent.mkdir(exist_ok=True)
    res_file_path.write_bytes(response.content)


if not file_path.exists():
    download_file(url, file_path)
