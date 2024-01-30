import time
from pathlib import Path
from argparse import ArgumentParser

from ir_dataset_utils import DASHED_DATASET_MAP
from tira.rest_api_client import Client
from tqdm import tqdm


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--softwares", nargs="+", default=None)

    args = parser.parse_args(args)

    tira = Client()

    TASK = "ir-benchmarks"
    all_softwares = tira.all_softwares(TASK)
    softwares_map = {}
    for software in all_softwares:
        split = software.split("/")[-1].split(" ")
        if len(split) == 1:
            key = split[0]
        else:
            key = "-".join(split[:-1])
        softwares_map[key.lower()] = software

    if args.softwares is None:
        softwares = softwares_map
    else:
        softwares = {}
        for software in args.softwares:
            if software in softwares_map:
                softwares[software] = softwares_map[software]
            else:
                raise ValueError(f"Software {software} not found.")

    settings = {}
    for key, software in tqdm(softwares.items(), desc="Getting settings"):
        software_settings = tira.docker_software(software)
        time.sleep(0.1)
        # if not software_settings["ir_re_ranker"]:
        #     continue
        settings[key] = software_settings

    for dataset in tqdm(tira.datasets(TASK).keys(), desc="Getting runs"):
        for software, software_settings in settings.items():
            software_settings = settings[software]
            dashed_dataset_id = "-".join(dataset.split("-")[:-2])
            try:
                dataset_id = DASHED_DATASET_MAP[dashed_dataset_id]
            except KeyError:
                dashed_dataset_id = dashed_dataset_id.replace(
                    "argsme-", "argsme-2020-04-01-"
                )
                try:
                    dataset_id = DASHED_DATASET_MAP[dashed_dataset_id]
                except:
                    continue
            dataset_id = dataset_id.replace("/", "-")
            target_dir = args.output_dir / "tirex" / software
            target_path = target_dir / f"{dataset_id}.run"
            target_path.unlink(missing_ok=True)
            software_name = softwares[software]
            if "obsolete" in software_name:
                # bug in tira client
                software_name = software_name[:-11]
            try:
                src_dir = Path(tira.get_run_output(software_name, dataset))
            except Exception as e:
                print(software, dataset)
                continue
            time.sleep(0.1)
            src_path = src_dir / "run.txt"
            if not src_path.exists():
                print(software, dataset)
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(src_path)


if __name__ == "__main__":
    main()
