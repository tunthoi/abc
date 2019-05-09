/* SPDX-License-Identifier: LGPL-2.1+ */
// Added by Hai Son
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dllmain.cpp : DLL�� �ʱ�ȭ ��ƾ�� �����մϴ�.
#include "stdafx.h"
#include <afxwin.h>
#include <afxdllx.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

static AFX_EXTENSION_MODULE abcDLL = { NULL, NULL };

extern "C" int APIENTRY
DllMain( HINSTANCE hInstance, DWORD dwReason, LPVOID lpReserved )
{
	// lpReserved�� ����ϴ� ��� ������ �����Ͻʽÿ�.
	UNREFERENCED_PARAMETER( lpReserved );

	if ( dwReason == DLL_PROCESS_ATTACH )
	{
		TRACE0( "abc.dll�ʱ�ȭ�ϰ� �ֽ��ϴ�.\n" );
		
		// Ȯ�� DLL�� �� ���� �ʱ�ȭ�մϴ�.
		if ( ! AfxInitExtensionModule( abcDLL, hInstance ) )
			return 0;

		// �� DLL�� ���ҽ� ü�ο� �����մϴ�.
		// ����: �� Ȯ�� DLL�� MFC ���� ���α׷��� �ƴ� ActiveX ��Ʈ�Ѱ� ���� MFC �⺻ DLL��
		//  ���� ��������� ��ũ�Ǿ� �ִ� ��쿡�� DllMain���� �� ���� �����ϰ�, ������ ���� �� Ȯ�� DLL����
		//  ������ ������ �Լ��� �߰��մϴ�.
		//  �׷� ���� �� Ȯ�� DLL�� ����ϴ� �⺻ DLL�� �ش� �Լ��� ��������� ȣ���Ͽ� �� Ȯ�� DLL�� �߰��ؾ� �մϴ�.
		//  �׷��� ������ CDynLinkLibrary ��ü�� �⺻ DLL�� ���ҽ� ü�ο� �߰����� �����Ƿ�
		//  �ɰ��� ������ �߻��մϴ�.

		new CDynLinkLibrary( abcDLL );
	}
	else if ( dwReason == DLL_PROCESS_DETACH )
	{
		TRACE0( "CXAlg.DLL�� �����ϰ� �ֽ��ϴ�.\n" );

		// �Ҹ��ڰ� ȣ��Ǳ� ���� ���̺귯���� �����մϴ�.
		AfxTermExtensionModule( abcDLL );
	}
	return 1;   // Ȯ��
}
