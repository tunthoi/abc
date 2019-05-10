/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once 

namespace comed { namespace abc 
{
	/// <summary>
	/// ABC feature id
	/// </summary>
	enum E_ABCFeatureId 
	{
		kABCFeatureId_Global_Otsu = 0,
		kABCFeatureId_Global_Max,
		kABCFeatureId_Global_Min,
		kABCFeatureId_Global_Mean,
		kABCFeatureId_Global_Std,
		kABCFeatureId_Global_Mode,
		kABCFeatureId_Global_KV,
		kABCFeatureId_Global_MA,

		kABCFeatureId_Local_Otsu,
		kABCFeatureId_Local_Max,
		kABCFeatureId_Local_Min,
		kABCFeatureId_Local_Mean,
		kABCFeatureId_Local_Std,
		kABCFeatureId_Local_Mode,

		_END_ABC_Features
	};

	/// <summary>
	/// ABC result id
	/// </summary>
	enum E_ABCResultId
	{
		kABCResultId_Metal = 0,
		kABCResultId_Background,

		_END_ABC_Results
	};

	/// <summary>
	/// classfier result
	/// </summary>
	struct RegionType
	{
		bool bMetal, bBackground;
	};

}} // comed::abc

// macro constant 
// 16x16 Slice Area
#define ABC_REGION_DIVIDE					16
#define ABC_REGION_DIVIDE_2					( ABC_REGION_DIVIDE * ABC_REGION_DIVIDE )

#define ABC_FEATURE_COUNT					( comed::abc::_END_ABC_Features )
#define ABC_RESULT_COUNT					( comed::abc::_END_ABC_Results )

#define ABC_MIN_TRAININGDATACOUNT			10

