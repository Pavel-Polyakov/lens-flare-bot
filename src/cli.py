import argparse
import logging

from lens_flare import StepByStepLensFlare, Config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-s', '--star', type=str, default='star.png')


    def _get_output(input_file):
        prefix, postfix = input_file.split('.')
        return f'{prefix}_stars.{postfix}'


    args = parser.parse_args()

    cfg = Config(
        src_file=args.input,
        out_file=args.output or _get_output(args.input),
        star_file=args.star,
        threshold_min_percent=99.5,
        threshold_max_percent=100.0,
        lights_size_min=50,
        lights_size_max=100,
        star_multiplier=10,
    )

    logger = logging.getLogger(__name__)

    iterator = StepByStepLensFlare(cfg, logger).process()
    skip = None
    while True:
        try:
            out_file = iterator.send(skip)
        except StopIteration:
            break
        print(f'intermediate file {out_file}')
        import subprocess

        subprocess.call(['open', out_file])
        skip_str = input('should we save it? n or y')
        if skip_str != 'y':
            skip = True
        else:
            skip = False
