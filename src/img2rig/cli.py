"""Command-line entry point.

    img2rig <command> --spec character.yaml [args]

Commands mirror the pipeline stages: gen, finalize, variant, keyframes,
retouch (generation, this module tree) and split, cleanup, export (layering /
refinement / packaging, loaded lazily - they may ship separately).
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module

from . import config, gen, keyframes, retouch, variant
from .sdapi import Client


def _client(spec: config.Spec) -> Client:
    return Client(spec.api, spec.sd)


def _stage(name: str):
    """split/cleanup/export import heavier deps (cv2, pytoshop); load them at
    call time so the generation commands work even without those installed."""
    try:
        return import_module(f"img2rig.{name}")
    except ImportError as e:
        sys.exit(f"img2rig: the '{name}' stage is not available ({e})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="img2rig", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--spec", required=True, help="character spec YAML")
        return p

    add("gen", "roll txt2img candidates + contact sheet")

    p = add("finalize", "re-run a picked seed with hires-fix")
    p.add_argument("--seed", type=int, required=True, help="picked candidate seed")
    p.add_argument("--upscale", type=float, default=None, metavar="SCALE",
                   help="also GAN-upscale the final by SCALE (e.g. 2.0)")

    p = add("variant", "img2img state variant over a source image")
    p.add_argument("name", help="variant name from the spec's variants section")
    p.add_argument("--src", required=True, help="source image (hires final)")
    p.add_argument("--seed", type=int, required=True,
                   help="seed (normally the final's seed)")

    p = add("keyframes", "roll pose keyframe candidates (two routes)")
    p.add_argument("--base", required=True, help="base image (hires final)")

    p = add("retouch", "inpaint one region of an image")
    p.add_argument("--src", required=True, help="image to retouch")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="rectangle to re-roll, pixel coords")
    g.add_argument("--mask", help="grayscale mask image (white = editable)")
    p.add_argument("--add", default="", help="extra positive tags for the region")
    p.add_argument("--denoise", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=0)

    p = add("split", "first-pass layering (DINO + SAM)")
    p.add_argument("--hires", required=True, help="picked hires final image")

    p = add("cleanup", "mask refinement and layer assembly")
    p.add_argument("--hires", required=True, help="picked hires final image")
    p.add_argument("--base4k", help="4K layer source (defaults to --hires)")
    p.add_argument("--variant4k", help="4K state variant for glow derivation")
    p.add_argument("--stage", default="all",
                   choices=["face", "body", "hair", "glow", "fill",
                            "assemble", "qa", "all"],
                   help="run one refinement pass (iterate the visual loop) "
                        "or the whole chain")

    p = add("export", "pack layers + rig.txt for the runtime")
    p.add_argument("--base4k", required=True, help="4K layer source")
    p.add_argument("--variant4k", help="4K state variant (glow layer pixels)")
    p.add_argument("--kf", nargs=2, action="append", metavar=("FRAME", "IMG"),
                   help="also matte a pose keyframe: frame name + picked "
                        "candidate image (repeatable)")
    p.add_argument("--base", help="base image for keyframe diff points "
                                  "(required with --kf)")

    p = add("preview", "render an animated GIF/WebP preview of the exported rig")
    p.add_argument("--script", default="idle", choices=["idle", "showcase"])
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default=None, help="output filename (in work_dir)")

    args = ap.parse_args(argv)
    spec = config.load(args.spec)
    client = _client(spec)

    if args.cmd == "gen":
        gen.candidates(spec, client)
    elif args.cmd == "finalize":
        p_final = gen.finalize(spec, client, args.seed)
        if args.upscale:
            gen.upscale_4k(spec, client, p_final, scale=args.upscale)
    elif args.cmd == "variant":
        variant.make(spec, client, args.name, args.src, args.seed)
    elif args.cmd == "keyframes":
        keyframes.candidates(spec, client, args.base)
    elif args.cmd == "retouch":
        region = tuple(args.box) if args.box else args.mask
        retouch.inpaint_region(spec, client, args.src, region,
                               prompt_add=args.add, denoise=args.denoise,
                               seed=args.seed)
    elif args.cmd == "split":
        _stage("split").draft(spec, client, args.hires)
    elif args.cmd == "cleanup":
        cleanup = _stage("cleanup")
        base4k = args.base4k or args.hires
        st = args.stage
        if st in ("face", "all"):
            cleanup.face_parts(spec, client, args.hires)
        if st in ("body", "all"):
            cleanup.body_parts(spec, client, args.hires)
        if st in ("hair", "all"):
            cleanup.hair_split(spec, args.hires)
        if st in ("glow", "all") and args.variant4k:
            cleanup.derive_glow(spec, base4k, args.variant4k)
        filled = None
        if st in ("fill", "assemble", "qa", "all"):
            filled = cleanup.fill_occlusion(spec, base4k)
        if st in ("assemble", "qa", "all"):
            layers = cleanup.assemble(spec, base4k, args.variant4k, filled)
            if st in ("qa", "all"):
                cleanup.qa_sheets(spec, layers, base4k)
    elif args.cmd == "export":
        exp = _stage("export")
        cleanup = _stage("cleanup")
        layers = cleanup.assemble(spec, args.base4k, args.variant4k, None)
        exp.layer_pack(spec, layers)
        if args.kf:
            if not args.base:
                sys.exit("img2rig export: --kf requires --base")
            for frame, img in args.kf:
                exp.keyframe(spec, client, frame, img, args.base)
    elif args.cmd == "preview":
        _stage("preview").run(spec, script=args.script, out_name=args.out,
                              height=args.height, fps=args.fps)


if __name__ == "__main__":
    main()
