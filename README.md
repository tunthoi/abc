# abc
fluoroscopic automatic brightness control
##	Scope
본 문서는 GEMSS Medical Systems 사의
 SPINEL-12HD 의 ODDC 를 위한 Object Detection S/W의 파라메터 작성
을 위하여 작성되었다.

##	Preparation(준비물)
디텍터 캘리브레이션이 완료된 장비 1 Set
인체 팬텀 : Skull, Chest, Pelvis, Knee, Hand, Foot 각 1 set
침대: 아크릴, 카본 각 1 set
그리드 1 set
수술용 도구( 집게, 나이프, 나사못 등 )

##	영상획득 과정
1.	시스템을 부팅한다.
2.	‘연속촬영’, ‘ABS OFF’, ‘Auto Capture ON’ 모드로 조정한다.
3.	팬텀이 영상의 중앙에 출력되도록 침대위에 배치한다. 이때 C-arm 의 위치는 일반 촬영과 동일하게 침대의 아래에 X-ray 소스가 오게 한다.
4.	40 kV 부터 120 kV 까지 5 kV 간격으로 영상을 촬영 저장한다.
5.	팬텀을 움직이지 않고 그대로 두고 수술용 도구를 적절한 위치에 배치한다.
6.	40 kV 부터 120 kV 까지 5 kV 간격으로 영상을 촬영 저장한다.
7.	팬텀의 자세를 바꾸거나 종류를 바꿔서 4-6번까지의 과정을 재실행 한다.
8.	그리드를 사용하는 경우 그리드를 끼우고 4-6번까지의 과정을 반복한다.
9.	시스템을 종료한다.

##	Ground Truth(진실값) 입력하기
1.	시스템을 부팅한다.
2.	Admin 계정으로 로그인 한다.
3.	파라메터 작성용 영상이 촬영된 환자를 선택하여 Open 한다.
4.	Reference thumbnail view 에서 ground truth 를 작성할 파일을 선택한다.
5.	‘t’를 입력하여 Ground truth 입력 모드로 진입한다.
6.	‘b’, ‘o’ key 와 마우스 왼쪽버튼을 이용하여 ‘background’ 와 ‘object’를 표시한다. 이때 ‘object’와 ‘background’가 공존하는 경우에는 ‘background’로 표시한다.
7.	‘m’, ‘n’ 를 이용하여 ‘metal’ 과 ‘non-metal’ 을 표시한다. 이때 ‘metal’ 과 ‘non-metal’ 이 공존하는 경우에는 ‘non-metal’ 로 표시한다.
8.	‘s’ 나 ‘c’ key 를 이용하여 ground truth 를 저장한다.
9.	Ground truth 를 입력할 파일이 더 있을 경우 4번-8번을 반복한다.
10.	프로그램을 종료한다.

##	파라메터 작성하기
1.	Cxview 프로그램 디렉토리에 있는 ‘abc-trainner.exe’를 실행한다.
2.	‘…’ 를 클릭하여 Ground truth 를 입력한 환자의 데이터 디렉토리를 지정한다.
3.	‘OK’ 버튼을 클릭한다.
4.	파라메터 작성이 완성되면 ‘Cancel’을 눌러 프로그램을 종료한다.
5.	파라메터 지정했던 환자의 데이터 디렉토리에 ‘abc.training.result.metal’ 과 ‘abc.training.result.objec’ 파일로 생성된다.

##	파라메터 적용하기
1.	‘abc.training.result.metal’ 과 ‘abc.trainning.result.objec’ 파일을 CXview 3.0 의 Configuration 디렉토리에 복사한다.

7	파라메터 테스트
1.	시스템을 부팅한다.
2.	Admin 계정으로 로그인 한다
3.	촬영 모드로 진입한다.
4.	ABC 기능을 ON 시킨다
5.	‘a’ key 로 ODDC 의 결과가 화면에 표시되도록 한다.
6.	Continuous 모드로 촬영하면서 ODDC 결과를 확인한다.
