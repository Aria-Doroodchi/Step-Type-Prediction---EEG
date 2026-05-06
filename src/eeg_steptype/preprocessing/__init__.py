"""Raw .bdf → cleaned epochs .fif.

Pipeline stages (each in its own module):

    montage   constants for the 64-channel BioSemi → 10-20 mapping
    load      raw assembly: single file or concat-with-crops via override
    bads      automated bad-channel detection (PyPREP)
    asr       transient burst correction before ICA
    filter    ZapLine line-noise removal + bandpass
    reference provisional CAR for ICA + final CSD
    ica       fit + auto-classify (ICLabel) + apply
    events    extract condition-paired response events
    epoching  build epochs around response events
    reject    AutoReject-local epoch repair/rejection
    pipeline  orchestrator: ties them together for one participant
"""
