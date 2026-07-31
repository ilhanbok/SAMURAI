# __1.2.0__ __2026-07-30__
  - ### New features
    - Web portal (`webapp/`, `run_web.sh` / `run_web.ps1`): both modes of
      operation, the utilities and the workflow notes in a browser, backed by
      the same Python pipeline
    - Processing modules split out headless (`webapp/core/`), so the pipeline can
      be imported without a matplotlib/Qt display
    - Per-session working directories with download links for every output
  - ### Fixes
    - Threshold summary CSV no longer drops the highest label (the desktop
      wrappers write `range(0, num_features)`); one lesion per scan was lost
    - Web `DUPLICATE` takes an output path; the CLI version referenced an
      undefined `output_file` and could not run
    - The annotated slice is the slice SAM2 is prompted on; set
      `SAMURAI_ITKSNAP_OFFSET=1` for the previous `slice - 1` behaviour

# __1.1.0__ __2025-06-19__
  - ### New features
    - Fixed mask offset
    - Axial segmentation prediction renders in all three dimensions

# __1.0.0__ __2025-04-02__
 - Initial public release of software as tested by Dr. Kanter
   
