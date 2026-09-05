"""Decoder-only U-Net, gated ELU convs, camera/tone-mapper model, losses.

Module: trippy.net
Invariants: (none yet -- empty stub)
Related docs: /tmp/trippy-plan.md "Verified facts" (TRIPS default net:
    decoder-only U-Net, 5 levels, 32 filters, gated ELU convs, ~130k params;
    per-image exposure + response LUT tone mapper) and "Technical design"
    (L1 + SSIM + LPIPS/VGG16 losses).
"""
