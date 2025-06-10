import os, json
import argparse
import pandas as pd
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--metrics_dir', type=str, required=True) # metrics and hyperparams ablations
    parser.add_argument('--model', type=str, required=True) # used model
    args = parser.parse_args()

#     metadata = {
#     'config1': {'metric1': 100, 'metric2': 200},
#     'config2': {'metric1': 110, 'metric3': 210},
#     'config3': {'metric2': 220, 'metric3': 230}
# }
#     df = pd.DataFrame.from_dict(metadata, orient='index')
    
    # for every model
    metadata = {}
    for file in os.listdir(args.metrics_dir):
        with open(file) as f:
            metrics = json.load(f)
            metadata[file.split("/")[-1]] = metrics

    metadata = {
    'config1': {'metric1': 100, 'metric2': 200},
    'config2': {'metric1': 110, 'metric3': 210},
    'config3': {'metric2': 220, 'metric3': 230}
}
    df = pd.DataFrame.from_dict(metadata, orient='index')
    

    df.to_csv(os.path.join(os.path.dirname(args.metrics_dir), f"{args.model}.csv"), index_label='Configuration', na_rep='N/A')

    