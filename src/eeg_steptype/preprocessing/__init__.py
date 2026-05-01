"""Raw .bdf → cleaned epochs .fif.

Pipeline stages (each in its own module):

    montage   constants for the 64-channel BioSemi → 10-20 mapping
    load      raw assembly: single file or concat-with-crops via override
    bads      automated bad-channel detection (PyPREP)
    filter    notch + bandpass
    reference average reference projection
    ica       fit + auto-classify (ICLabel) + apply
    events    extract condition-paired response events
    epoching  build epochs around response events
    reject    automated artifact rejection (autoreject), with threshold fallback
    pipeline  orchestrator: ties them together for one participant
"""
