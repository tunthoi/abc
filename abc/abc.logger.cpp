/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "abc.logger.h"


// NOTE by Hai Son
// If you put the Log4cxx.lib in the project settings as input, it will be linked before the MFC library.
// In this case, a number of false memory leakage detection report occurs in debug version.
// If you link the library by using a pragma statement, it is linked later than the MFC library.
#pragma comment( lib, "log4cxx.lib" )


DEFINE_MODULE_LOGGER( cxview3, cxview3.abc )


