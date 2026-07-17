#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shutil
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int], None]
DoneCb = Callable[[bool, Optional[str], Optional[str]], None]


class CharmRunner:
    """Runs SimNIBS CHARM segmentation via the Python API in a background thread.

    Callbacks (called from the worker thread):
      progress_cb(message: str, percent: int)
      done_cb(success: bool, error: str | None, subject_dir: str | None)
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        subject_dir: str,
        mri_files: list[str],
        forcerun: bool = False,
        force_qform: bool = False,
        force_sform: bool = False,
        progress_cb: ProgressCb = lambda *_: None,
        done_cb: DoneCb = lambda *_: None,
    ) -> None:
        self._thread = threading.Thread(
            target=self._run,
            args=(subject_dir, mri_files, forcerun, force_qform, force_sform, progress_cb, done_cb),
            daemon=True,
            name="charm-runner",
        )
        self._thread.start()

    def _run(
        self,
        subject_dir: str,
        mri_files: list[str],
        forcerun: bool,
        force_qform: bool,
        force_sform: bool,
        progress_cb: ProgressCb,
        done_cb: DoneCb,
    ) -> None:
        try:
            from simnibs.segmentation import charm_main
        except ImportError as exc:
            done_cb(False, f"Could not import simnibs: {exc}", None)
            return

        if not mri_files:
            done_cb(False, "No MRI files provided.", None)
            return

        t1 = mri_files[0]
        t2 = mri_files[1] if len(mri_files) > 1 else None

        for path in mri_files:
            if not os.path.isfile(path):
                done_cb(False, f"MRI file not found: {path}", None)
                return

        if forcerun and os.path.isdir(subject_dir):
            log.info("forcerun: removing existing directory %s", subject_dir)
            try:
                shutil.rmtree(subject_dir)
            except Exception as exc:
                done_cb(False, f"forcerun: could not remove {subject_dir}: {exc}", None)
                return

        os.makedirs(subject_dir, exist_ok=True)
        progress_cb("Starting CHARM segmentation…", 0)

        try:
            charm_main.run(
                subject_dir=subject_dir,
                T1=t1,
                T2=t2,
                registerT2=bool(t2),
                initatlas=True,
                segment=True,
                create_surfaces=True,
                mesh_image=True,
                force_qform=force_qform,
                force_sform=force_sform,
            )
        except Exception as exc:
            done_cb(False, str(exc), None)
            return

        if not os.path.isdir(subject_dir):
            done_cb(False, f"Expected output folder not found: {subject_dir}", None)
            return

        progress_cb("Head model complete.", 100)
        done_cb(True, None, subject_dir)
