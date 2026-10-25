from json import dumps, loads
from pathlib import Path
from sys import argv

from loguru import logger
from rich.progress import track
from torch import accelerator, device
from wtpsplit import SaT


def main():
    model = SaT("sat-3l-sm")

    dev = accelerator.current_accelerator(check_available=True) or device("cpu")
    logger.info(f"Using device: {dev}")
    if dev.type == "cpu":
        model.to(dev)
    else:
        model.half().to(dev)

    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            continue

        if path.is_dir():
            file_paths = []

            for root, _, files in path.walk():
                for file in files:
                    file_path = root / file
                    if file_path.suffix.lower() in (".txt"):
                        file_paths.append(file_path)

            for file_path in track(file_paths):
                item = model.split(file_path.read_text())

                splitted_path = file_path.with_suffix(".json")
                splitted_path.write_text(dumps(item, ensure_ascii=False))
        else:
            items = loads(path.read_bytes())
            for item in track(items):
                item["sentences"] = model.split(item["content"])

            splitted_path = path.with_stem(f"{path.stem}.splitted")
            splitted_path.write_text(dumps(items, ensure_ascii=False))


main()
