"""
Deposit METEOR emulator artifacts to Zenodo.

The repository is already linked to Zenodo through the GitHub integration, but
that integration archives only the repository source archive at each release.
It cannot publish separately-built artifacts, so bundles and golden fixtures
need their own deposit -- as a *dataset*, distinct from the software record.

The awkward part is ordering: an artifact should record the DOI of the deposit
it belongs to, but the DOI does not exist until the deposit does. Zenodo solves
this with DOI pre-reservation, so the workflow is three steps rather than one:

    1. reserve     create a draft deposit and print its reserved DOI
    2. (rebuild)   regenerate artifacts with doi=<that DOI> embedded
    3. upload      attach the rebuilt files to the draft
    4. publish     mint the DOI permanently

Steps are separate commands on purpose: each one is auditable, and publishing
is irreversible.

Usage
-----
    export ZENODO_TOKEN=...            # scopes: deposit:write, deposit:actions

    python scripts/deposit_bundles.py reserve --title "..." --version v1
    python scripts/deposit_bundles.py upload  <id> dist/*.nc
    python scripts/deposit_bundles.py publish <id> --yes

Defaults to the Zenodo *sandbox*. Pass --production to touch the real archive;
that flag is required for anything that mints a real DOI.

Notes
-----
Requires ``requests``, which is not a METEOR dependency -- this is an
operational script, not library code, and is not imported by the package.
"""

import argparse
import hashlib
import json
import os
import sys

SANDBOX_API = "https://sandbox.zenodo.org/api"
PRODUCTION_API = "https://zenodo.org/api"

#: Concept DOI of the METEOR software record, cross-linked from the dataset so
#: the two are navigable in both directions.
SOFTWARE_CONCEPT_DOI = "10.5281/zenodo.14967115"

DEFAULT_DESCRIPTION = """
<p>Compact per-location emulator bundles produced by
<a href="https://github.com/benmsanderson/METEOR">METEOR</a>, for clients that
need regional or point time series without the gridded model.</p>

<p>Each bundle is a self-describing netCDF file holding the VARX innovation
arrays, per-location seasonal and EOF projections, the pattern-scaling
step-response kernel, and pre-computed forcing for a set of SSP scenarios. The
schema is documented in
<code>docs/emulator_artifact_schema.md</code> in the source repository.</p>

<p>Golden fixtures provide fixed-seed reference output for validating a
reimplementation in another language.</p>
""".strip()


def _require_requests():
    """
    Import requests, with an actionable message when it is absent.

    Returns
    -------
    module
        The imported ``requests`` module.
    """
    try:
        import requests  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        sys.exit(
            "This script needs 'requests', which METEOR does not depend on.\n"
            "Install it just for this: pip install requests"
        )
    return requests


def _token():
    """
    Read the Zenodo access token from the environment.

    Returns
    -------
    str
        The token. Never logged.
    """
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit(
            "ZENODO_TOKEN is not set.\n"
            "Create one at Zenodo > Settings > Applications > Personal access "
            "tokens, with scopes 'deposit:write' and 'deposit:actions'."
        )
    return token


def _api(args):
    """
    Resolve the API base URL for this invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments carrying ``production``.

    Returns
    -------
    str
        Base URL.
    """
    return PRODUCTION_API if args.production else SANDBOX_API


def _headers():
    """
    Build the auth header.

    Returns
    -------
    dict
        Authorization header.
    """
    return {"Authorization": f"Bearer {_token()}"}


def _check(response, what):
    """
    Fail loudly on an unsuccessful API response.

    Parameters
    ----------
    response : requests.Response
        Response to check.
    what : str
        Human-readable description of the attempted operation.

    Returns
    -------
    dict
        Parsed JSON body, or an empty dict for empty responses.
    """
    if response.status_code >= 400:
        sys.exit(f"{what} failed: HTTP {response.status_code}\n{response.text[:800]}")
    if not response.content:
        return {}
    return response.json()


def _artifact_attrs(path):
    """
    Read provenance attributes from an artifact, if it is readable as netCDF.

    Parameters
    ----------
    path : str
        Path to a candidate artifact.

    Returns
    -------
    dict
        Attributes, or an empty dict when the file is not a readable dataset.
    """
    try:
        import xarray as xr  # pylint: disable=import-outside-toplevel

        with xr.open_dataset(path) as dataset:
            return dict(dataset.attrs)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def cmd_reserve(args):
    """
    Create a draft deposit and print its pre-reserved DOI.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    requests = _require_requests()
    metadata = {
        "upload_type": "dataset",
        "title": args.title,
        "description": DEFAULT_DESCRIPTION,
        "creators": [{"name": name} for name in args.creator],
        "version": args.version,
        "license": args.license,
        "prereserve_doi": True,
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": SOFTWARE_CONCEPT_DOI,
                "scheme": "doi",
            }
        ],
    }
    response = requests.post(
        f"{_api(args)}/deposit/depositions",
        headers={**_headers(), "Content-Type": "application/json"},
        data=json.dumps({"metadata": metadata}),
        timeout=60,
    )
    body = _check(response, "Creating deposition")
    reserved = body.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    print(f"deposition id : {body['id']}")
    print(f"reserved DOI  : {reserved}")
    print(f"draft         : {body.get('links', {}).get('html', '')}")
    print(f"target        : {'PRODUCTION' if args.production else 'sandbox'}")
    print()
    print("Next: rebuild the artifacts with this DOI embedded, e.g.")
    print(f'    export_timeseries_bundle(..., doi="{reserved}")')
    print(f"Then: {sys.argv[0]} upload {body['id']} <files...>")


def cmd_upload(args):
    """
    Upload files to an existing draft deposit and verify their checksums.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    requests = _require_requests()
    base = _api(args)
    deposition = _check(
        requests.get(
            f"{base}/deposit/depositions/{args.deposition_id}",
            headers=_headers(),
            timeout=60,
        ),
        "Fetching deposition",
    )
    bucket = deposition.get("links", {}).get("bucket")
    if not bucket:
        sys.exit(
            f"Deposition {args.deposition_id} exposes no bucket link; it may "
            "already be published. Use 'newversion' to add files to a published "
            "record."
        )
    reserved = deposition.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")

    for path in args.files:
        if not os.path.isfile(path):
            sys.exit(f"Not a file: {path}")
        attrs = _artifact_attrs(path)
        version = attrs.get("meteor_version", "")
        embedded = attrs.get("doi", "")
        if "dirty" in version:
            print(
                f"  ! {os.path.basename(path)}: built from a dirty tree "
                f"({version}); it cannot be reproduced from any commit"
            )
        if reserved and embedded and embedded != reserved:
            print(
                f"  ! {os.path.basename(path)}: embedded doi {embedded!r} does "
                f"not match this deposit's reserved doi {reserved!r}"
            )
        if reserved and not embedded:
            print(
                f"  ! {os.path.basename(path)}: no doi recorded; rebuild with "
                f'doi="{reserved}" to make it self-identifying'
            )

        with open(path, "rb") as handle:
            payload = handle.read()
        local_md5 = hashlib.md5(payload).hexdigest()  # nosec - Zenodo's checksum
        name = os.path.basename(path)
        result = _check(
            requests.put(
                f"{bucket}/{name}",
                headers=_headers(),
                data=payload,
                timeout=600,
            ),
            f"Uploading {name}",
        )
        remote = str(result.get("checksum", "")).replace("md5:", "")
        status = "ok" if remote == local_md5 else f"CHECKSUM MISMATCH ({remote})"
        print(f"  uploaded {name}  {len(payload)/1024:.1f} KB  {status}")
        if remote != local_md5:
            sys.exit("Aborting: uploaded bytes do not match local file.")

    print(f"\nNext: {sys.argv[0]} publish {args.deposition_id} --yes")


def cmd_publish(args):
    """
    Publish a draft deposit, minting its DOI permanently.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    requests = _require_requests()
    if not args.yes:
        sys.exit(
            "Publishing is irreversible: the DOI is minted and the files "
            "become permanent. Re-run with --yes once you are sure."
        )
    if args.production:
        print("Publishing to PRODUCTION Zenodo.")
    body = _check(
        requests.post(
            f"{_api(args)}/deposit/depositions/{args.deposition_id}/actions/publish",
            headers=_headers(),
            timeout=120,
        ),
        "Publishing deposition",
    )
    print(f"published : {body.get('doi_url', body.get('doi', ''))}")
    print(f"record    : {body.get('links', {}).get('record_html', '')}")


def cmd_newversion(args):
    """
    Open a new draft version of an already published deposit.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    requests = _require_requests()
    body = _check(
        requests.post(
            f"{_api(args)}/deposit/depositions/"
            f"{args.deposition_id}/actions/newversion",
            headers=_headers(),
            timeout=120,
        ),
        "Creating new version",
    )
    draft = body.get("links", {}).get("latest_draft", "")
    print(f"new draft : {draft}")
    print(f"draft id  : {draft.rstrip('/').split('/')[-1] if draft else '?'}")


def build_parser():
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="target real Zenodo instead of the sandbox (required to mint a real DOI)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reserve = subparsers.add_parser("reserve", help="create a draft and reserve a DOI")
    reserve.add_argument(
        "--title", default="METEOR emulator bundles", help="deposit title"
    )
    reserve.add_argument(
        "--creator",
        action="append",
        default=None,
        help='creator as "Last, First"; repeatable',
    )
    reserve.add_argument("--version", default="v1", help="deposit version label")
    reserve.add_argument("--license", default="cc-by-4.0", help="Zenodo license id")
    reserve.set_defaults(func=cmd_reserve)

    upload = subparsers.add_parser("upload", help="attach files to a draft")
    upload.add_argument("deposition_id")
    upload.add_argument("files", nargs="+")
    upload.set_defaults(func=cmd_upload)

    publish = subparsers.add_parser("publish", help="publish a draft (irreversible)")
    publish.add_argument("deposition_id")
    publish.add_argument("--yes", action="store_true", help="confirm minting the DOI")
    publish.set_defaults(func=cmd_publish)

    newversion = subparsers.add_parser(
        "newversion", help="open a new version of a published deposit"
    )
    newversion.add_argument("deposition_id")
    newversion.set_defaults(func=cmd_newversion)
    return parser


def main(argv=None):
    """
    Entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.
    """
    args = build_parser().parse_args(argv)
    if getattr(args, "creator", None) is None and args.command == "reserve":
        args.creator = ["Sanderson, Benjamin M."]
    args.func(args)


if __name__ == "__main__":
    main()
