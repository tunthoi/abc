/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "clUtils/defines.h"
#include "abc/abc_types.h"

// forward declaration
namespace cl { namespace img { class CImageBuf; }}


namespace comed { namespace abc 
{
	/// <summary>
	/// ���� opencv�� �ڵ带 Ȱ���� feature generator�� �ӵ��� ���������� �ʾ� ���� �����ϴ� Ŭ����
	/// </summary>
	class CFeatureGen
	{
		CL_NO_INSTANTIATION( CFeatureGen );

		// public methods
	public:

		/// <summary>
		/// only public methods
		/// </summary>
		static bool CalcFeatures( 
				IN		const cl::img::CImageBuf & img, OUT		double **dbFeatures );

		// private methods
	private:

		// calculate local statistics
		static void _calcLocalStatistics( 
			const WORD* pwSrc, int nImgW, int nImgH, int nBlkW, int nBlkH, 
			WORD awLocalMax[], WORD awLocalMin[], UINT anLocalSum[], double adbLocalSoS[] );

		// calc one block statistics
		static void _calcBlockStatistics( 
			const WORD* pwBlk, int nStrider, int nBlkW, int nBlkH, 
			WORD* pwLocalMax, WORD* pwLocalMin, UINT* pnLocalSum, double* pdbLocalSoS );

		// calc otsu
		static void _calcOtsu( 
					const WORD* pwSrc, int nStrider, int nBlkW, int nBlkH, WORD wMin, WORD wMax,
					double* pdOtsu, double* pdInner, double* pdInter );
	};
}} // comed::abc
