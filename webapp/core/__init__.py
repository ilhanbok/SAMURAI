"""Headless ports of the SAMURAI processing modules.

Each module here mirrors one of the desktop scripts:

===========================  ==================================================
``volume``                   NIfTI IO / orientation / JPEG stack export
``render``                   slice + overlay PNG encoding (replaces imshow)
``segmenter``                SAM2 prompting and propagation (mode i, SAMURAI-BB)
``threshold``                RAW/QSM threshold lesion finder (mode ii)
``utilities``                MERGE / ZERO / DUPLICATE / IRV / VOLUME
``session``                  per-browser-session workspace and state
===========================  ==================================================
"""
