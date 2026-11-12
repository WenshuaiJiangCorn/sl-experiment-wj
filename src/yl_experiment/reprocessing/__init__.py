"""This package provides the tooling for re-processing the log archives recorded before the 2026-08-11 firmware fix.

Firmware predating that fix serialized the zero readout of the lick and analog modules through the explicit-prototype
SendData overload, passing a bare integer literal as the data object. The literal was deduced as a 32-bit int and
serialized as 4 bytes, while the message header declared the 2-byte uint16 prototype. The updated extraction pipeline
validates every payload against its declared prototype and refuses an archive holding such a message, which makes
every archive recorded before the fix unreadable. The modules of this package correct the affected archives, re-run
the extraction and parsing pipeline against them, and drive that pipeline across every session of a project.

The directory layout the pipeline expects:

    <session>/
        linear_track_data_log/      The raw .npz archives. Never modified.
        unsanitized_processed/      The parse produced before sanitization. Read as the seed.
        processed/                  The re-processed parse. Written by the pipeline, read by the analysis code.

Rename 'processed' to 'unsanitized_processed' before re-processing a project:

    The runtime writes its parse to 'processed', so a project that has not been re-processed yet holds the affected
    parse under that name. The batch driver seeds from 'unsanitized_processed' and writes to 'processed', so running
    it against a project still in the runtime layout finds no seed and writes over the only parse that exists.
    Renaming first is what separates the two, and it is the one step the driver cannot do for itself, as it cannot
    tell a parse that predates sanitization from one it produced.

    Rename every session's 'processed' directory to 'unsanitized_processed', then run the batch driver. The driver
    recreates 'processed' for each session, seeded from 'unsanitized_processed' so that the files it does not
    regenerate are carried over, and regenerates the lick and analog data from the sanitized archive.

Notes:
    The modules are ordered by the layer they occupy. The archive_sanitization module corrects a single archive, the
    reprocess_legacy_logs module turns one corrected archive into the parsed .feather files, the batch_reprocess
    module runs that pipeline across a whole project, and the watch_progress module renders the progress of a batch
    run. The promote_outputs module performs the rename described above.

    Valve data is never regenerated. Dispensed volumes are reconstructed from the valve's pulse durations through the
    calibration the runtime used, which the archives do not record, so the valve files of the earlier parse are
    carried over unchanged and remain the only correct copy that exists.

    The output directory is the one the analysis code reads, so a session is re-processed in place. Seeding copies
    the earlier parse over that directory before anything is regenerated, which means a forced re-run that fails
    partway leaves the read directory holding the earlier parse until the run is repeated. A session that is already
    complete is skipped before any seeding happens, so an ordinary resume never reaches that state.
"""
