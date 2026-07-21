#pragma once

// Hardware accelerator register field width limits.
// Validate these before writing to prevent undefined IP behavior (hardware freeze).
#define IFX_ACTIVATIONSTREAMERCHANNELTIMESWIDTH        0x1FFFFUL
#define IFX_ACTIVATIONSTREAMERKERNELCHANNELTIMESWIDTH  0x3FFFFUL
#define IFX_ACTIVATIONSTREAMERREPEATS                  0x3FFFFUL
#define IFX_STRIDE_STRIDECHANNELTIMESCOLUMN            0xFFFFUL

