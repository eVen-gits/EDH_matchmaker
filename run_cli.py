import argparse
from InquirerPy import inquirer
from src.core import Tournament

def run():
    while t.structure:
        action = inquirer.select(
            message="Select an action",
            choices=t.structure.actions,
        ).execute()
        action = getattr(t.structure, action)
        action()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLI')

    t = Tournament()
    run()