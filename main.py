from system.model import PowerSystem

if __name__ == "__main__":
    system = PowerSystem(
        {
            1: {"v": 1},
            2: {"v": 0.9}
        }
    )
    