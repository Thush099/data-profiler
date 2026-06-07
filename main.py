
import argparse
import logging
from src.pipeline import DataProfiler
from src.utils import load_data, save_report
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Data Profiler')
    parser.add_argument('--input-data', help='Input data file', required=True)
    parser.add_argument('--output-report', help='Output report file', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = load_data(args.input_data)
    profiler = DataProfiler(Config())
    report = profiler.profile(data)

    save_report(report, args.output_report)

if __name__ == '__main__':
    main()
