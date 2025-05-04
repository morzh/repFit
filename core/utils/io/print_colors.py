class PrintColors:
    """
    Description:
        This class helps with colored text printing. Example of usage: print(f'{PrintColors.BOLD}some text{PrintColors.END}')
    """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'