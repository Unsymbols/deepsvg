from concurrent import futures
import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import glob
import pandas as pd

import sys
import traceback
import pdb


from deepsvg.svglib.svg import SVG


def preprocess_svg(svg_file, output_folder, meta_data):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    svg = SVG.load_svg(svg_file)
    svg.fill_(False)
    svg.normalize()
    svg.zoom(0.9)
    svg.canonicalize()
    svg = svg.simplify_heuristic()

    svg.save_svg(os.path.join(output_folder, f"{filename}.svg"))

    len_groups = [path_group.total_len() for path_group in svg.svg_path_groups]

    meta_data[filename] = {
        "id": filename,
        "total_len": sum(len_groups),
        "nb_groups": len(len_groups),
        "len_groups": len_groups,
        "max_len_group": max(len_groups)
    }


# def main2(args):
#     md = {}
#     preprocess_svg("/home/sh/o/unsymbols/deepsvg/dataset/unsymbols/cyr_svg/0A66E__CYRILLIC_LETTER_MULTIOCULAR_O__DejaVuSans.svg", "/tmp/svgs_out", md)
#     print(md)
#     breakpoint()

def main(args):
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        svg_files = glob.glob(os.path.join(args.data_folder, "*.svg"))
        print(f"Found {len(svg_files)} SVG files to preprocess.")
        meta_data = {}

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [executor.submit(preprocess_svg, svg_file, args.output_folder, meta_data)
                                    for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    df = pd.DataFrame(meta_data.values())
    df.to_csv(args.output_meta_file, index=False)

    logging.info("SVG Preprocessing complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--data_folder", default=os.path.join("dataset", "svgs"))
    parser.add_argument("--output_folder", default=os.path.join("dataset", "svgs_simplified"))
    parser.add_argument("--output_meta_file", default=os.path.join("dataset", "svg_meta.csv"))
    parser.add_argument("--workers", default=4, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)

    try:
        main(args)
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
