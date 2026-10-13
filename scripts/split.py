from torch import accelerator, device
from wtpsplit import SaT


def main():
    model = SaT("sat-3l-sm")

    dev = accelerator.current_accelerator(check_available=True) or device("cpu")
    if dev.type == "cpu":
        model.to(dev)
    else:
        model.half().to(dev)


main()
