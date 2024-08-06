import os
import csv

class Logger:
    def __init__(self, logs_filename='logs'):
        current_dir = os.getcwd()
        log_dir = os.path.join(current_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logs_path = os.path.join(log_dir, logs_filename + '.csv')

    def log(self, data: dict):
        file_exists = os.path.isfile(self.logs_path)
        with open(self.logs_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def delete_logs(self, verbose=True):
        if os.path.exists(self.logs_path):
            os.remove(self.logs_path)
            if verbose: print(f"Deleted log file at: {self.logs_path}")
        else:
            if verbose: print(f"No log file found at: {self.logs_path}")
